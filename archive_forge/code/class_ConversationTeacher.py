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
class ConversationTeacher(FixedDialogTeacher):
    """
    This module provides access to data in the Conversations format.

    Subclasses ``FixedDialogTeacher`` for functionality and provides an
    implementation of ``setup_data()`` which iterates over datasets in the
    "Conversations" format. If your data is in the format below, use this class to
    handle file parsing for you.

    The data should be set up so that each dialogue instance (or, episode)
    occupies one line of valid JSON. The way the data is set up is as follows
    (with line breaks for readability):

    ::

        {
            'dialogue':[
                {'id':'modelx', 'text': 'hi'},
                {'id':'modely', 'text': 'hi back'},
                ...
            ]
        }

    Note that by default, dialogs are interpreted as being one-way.
    For example, consider this dialog:

    ::

        {
            'dialogue':[
                {'id':'modelx', 'text': X1},
                {'id':'modely', 'text': Y1},
                {'id':'modelx', 'text': X2},
                {'id':'modely', 'text': Y2},
                {'id':'modelx', 'text': X3},
                {'id':'modely', 'text': Y3},
            ]
        }

    A set of examples X1 => Y1, X2 => Y2, and X3 => Y3 will be generated,
    forming one episode. However, Y1 => X2 and Y2 => X3 are not created as
    separate examples by default.
    To change this behavior, you can set opt['label_turns']. The default
    value is 'secondspeaker' (i.e., the second speaker's utterances are
    used as labels), but 'firstspeaker' and 'both' are also options. In the
    case of 'both', two episodes are generated for each conversation.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if not shared:
            self.episodes = []
            self.num_exs = 0
            self.label_turns = opt.get('label_turns')
            if opt.get('conversationteacher_datafile') is not None:
                self._setup_data(opt.get('conversationteacher_datafile'))
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
        return Message(self.episodes[episode_idx][entry_idx])

    def _setup_data(self, path):
        logging.info('[loading data from json file into task:' + path + ']')
        self.episodes = []
        self.num_exs = 0
        eps = []
        conversations = Conversations(path)
        self.num_exs = 0
        for conv in conversations:
            if conv.context:
                warn_once('At least one of these conversations contains a context, which is not being used')
            turns = [t for t in conv.turns if t.get('id') != 'context']
            if len(turns) != len(conv.turns):
                warn_once('At least one of these conversations contains a context within the dialogue, which is being discarded')
            turns.insert(0, {'text': '__SILENCE__'})
            if self.label_turns in ['firstspeaker', 'both']:
                eps = self._get_ep_from_turns(turns[::2], turns[1::2])
                if eps:
                    self.episodes.append(eps)
                    self.num_exs += len(eps)
            if self.label_turns in ['secondspeaker', 'both']:
                eps = self._get_ep_from_turns(turns[1::2], turns[2::2])
                if eps:
                    self.episodes.append(eps)
                    self.num_exs += len(eps)

    def _get_ep_from_turns(self, xturns, yturns):
        eps = []
        for xturn, yturn in zip(xturns, yturns):
            turn = {}
            turn['text'] = xturn.get('text').strip()
            turn['labels'] = [yturn.get('text').strip()]
            turn['episode_done'] = False
            eps.append(turn)
        if eps:
            eps[-1]['episode_done'] = True
            return eps