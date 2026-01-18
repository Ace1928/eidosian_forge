import copy
from typing import List, Tuple, Optional, TypeVar
from parlai.core.agents import Agent, create_agent_from_shared
from parlai.core.image_featurizers import ImageLoader
from parlai.core.loader import load_teacher_module
from parlai.core.loader import register_teacher  # noqa: F401
from parlai.core.message import Message
from parlai.core.metrics import TeacherMetrics, aggregate_named_reports
from parlai.core.opt import Opt
from parlai.utils.conversations import Conversations
from parlai.utils.data import DatatypeHelper
from parlai.utils.misc import AttrDict, no_lock, str_to_msg, warn_once
from parlai.utils.distributed import get_rank, num_workers, is_distributed
import parlai.utils.logging as logging
from abc import ABC, abstractmethod
import concurrent.futures
from threading import Thread
import queue
import random
import time
import os
import torch
import json
import argparse
class ParlAIDialogTeacher(FixedDialogTeacher):
    """
    This module provides access to data in the ParlAI Text Dialog format.

    Subclasses ``FixedDialogTeacher`` for functionality and provides an
    implementation of ``setup_data()`` which iterates over datasets in the
    "ParlAI text" format. If your data is in the format below, use this class to
    handle file parsing for you.

    The way the data is set up is as follows:

    ::

        text:Sam went to the kitchen. <NEWL>
        Pat gave Sam the milk. <NEWL>
        Where is the milk? <TAB> labels:kitchen <TAB> reward:1
        <TAB> label_candidates:hallway|kitchen|bathroom
        text:Sam went to the hallway. <NEWL>
        Pat went to the bathroom. <NEWL>
        Where is the milk? <TAB> labels:hallway <TAB> reward:1
        <TAB> label_candidates:hallway|kitchen|bathroom <TAB> episode_done:True

    Lines 1-2 represent a single episode, with a different example on each line.
    The lines contain a query and a label for getting the question
    correct, and three label candidates.

    Since both of these examples are part of the same episode, the information
    provided in the first example is relevant to the query in the second
    example and therefore the agent must remember the first example in order to
    do well.

    In general dialog this format can contain any speech, not just QA pairs:

    ::

        text:Hi how's it going?<TAB>labels:It's going great. What's new?
        text:Well I'm working on a new project at work.<TAB>labels:Oh me too!
        text:Oh cool!<TAB>labels:Tell me about yours.

    etc.

    Note that dialogs are interpreted as being one-way. For example, consider
    this dialog:

    ::

        1 X1    Y1
        2 X2    Y2
        3 X3    Y3

    A set of examples X1 => Y1, X2 => Y2, and X3 => Y3 will be generated.
    However, Y1 => X2 and Y2 => X3 are not created as separate examples by
    default. This makes sense for some data (we don't need to train on the idea
    that "kitchen" should be followed by "Sam went to the hallway..." above),
    but for other datasets it may be helpful to add additional examples in the
    reverse direction ("Oh cool!" is a response to "Oh me too!" above).
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if not shared:
            self.episodes = []
            self.num_exs = 0
            if opt.get('parlaidialogteacher_datafile') is not None:
                self._setup_data(opt.get('parlaidialogteacher_datafile'))
        else:
            self.episodes = shared['episodes']
            self.num_exs = sum((len(e) for e in self.episodes))
        self.id = opt['task']
        self.reset()

    def share(self):
        """
        Share the episodes.
        """
        shared = super().share()
        shared['episodes'] = self.episodes
        return shared

    def num_examples(self):
        """
        Return the number of examples from the data.
        """
        return self.num_exs

    def num_episodes(self):
        """
        Return the number of episodes from the data.
        """
        return len(self.episodes)

    def get(self, episode_idx, entry_idx=None):
        """
        Get a specific example from the dataset.
        """
        return self.episodes[episode_idx][entry_idx]

    def _setup_data(self, path):
        logging.info(f'Loading ParlAI text data: {path}')
        self.episodes = []
        self.num_exs = 0
        eps = []
        with open(path, newline='\n', encoding='utf-8') as read:
            for line_no, line in enumerate(read, 1):
                msg = str_to_msg(line.rstrip('\n'))
                if msg and 'eval_labels' in msg:
                    raise ValueError(f"It looks like you've written eval_labels as a key in your data file. This is not appropriate; labels will be converted for you automatically. This is happening on Line {line_no} in {path}. The line is:\n\t{line}")
                if msg and 'text' not in msg:
                    raise ValueError(f'ParlaiDialogTeacher requires a "text" field in every entry, but one is missing in Line {line_no} in {path}. The line is:\n\t{line}')
                if msg and 'labels' not in msg:
                    raise ValueError(f'ParlaiDialogTeacher requires a "labels" field in every entry, but one is missing in Line {line_no} in {path}. The line is:\n\t{line}')
                if msg:
                    self.num_exs += 1
                    eps.append(msg)
                    if msg.get('episode_done', False):
                        self.episodes.append(eps)
                        eps = []
        if len(eps) > 0:
            eps[-1].force_set('episode_done', True)
            self.episodes.append(eps)
        if len(self.episodes) == 1 and line_no > 100:
            logging.error(f'The data in {path} looks like one very long episode. If this is intentional, you may ignore this, but you MAY have a bug in your data.')