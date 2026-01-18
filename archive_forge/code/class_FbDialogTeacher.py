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
class FbDialogTeacher(DialogTeacher):
    """
    This module provides access to data in the Facebook Dialog format.

    Subclasses ``DialogTeacher`` for functionality and provides an
    implementation of ``setup_data()`` which iterates over datasets in the
    "fbdialog" format. If your data is in the format below, use this class to
    handle file parsing for you.

    The way FB Dialog data is set up is as follows:

    ::

        1 Sam went to the kitchen.
        2 Pat gave Sam the milk.
        3 Where is the milk?<TAB>kitchen<TAB>1<TAB>hallway|kitchen|bathroom
        4 Sam went to the hallway.
        5 Pat went to the bathroom.
        6 Where is the milk?<TAB>hallway<TAB>1<TAB>hallway|kitchen|bathroom

    Lines 1-6 represent a single episode, with two different examples: the
    first example is lines 1-3, and the second is lines 4-6.

    Lines 1,2,4, and 5 represent contextual information.

    Lines 3 and 6 contain a query, a label, a reward for getting the question
    correct, and three label candidates.

    Since both of these examples are part of the same episode, the information
    provided in the first example is relevant to the query in the second
    example and therefore the agent must remember the first example in order to
    do well.

    In general dialog in this format can contain any speech, not just QA pairs:

    ::

        1 Hi how's it going?<TAB>It's going great. What's new?
        2 Well I'm working on a new project at work.<TAB>Oh me too!
        3 Oh cool!<TAB>Tell me about yours.

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
        self.opt = opt
        self.cloze = opt.get('cloze', False)
        if shared and 'cands' in shared:
            self.cands = shared['cands']
        else:
            self.cands = self.load_cands(opt.get('cands_datafile', None))
        super().__init__(opt, shared)

    def share(self):
        """
        Share the data and canidates.
        """
        shared = super().share()
        shared['cands'] = self.cands
        return shared

    def label_candidates(self):
        """
        Return the candidates.
        """
        return self.cands

    def load_cands(self, path):
        """
        Load a global fixed set of candidates.

        The candidates will be provided by the teacher for every example (the true
        labels for a specific example are also added to this set, so that it's possible
        to get the right answer).
        """
        if path is None:
            return None
        cands = []
        lines_have_ids = False
        cands_are_replies = False
        cnt = 0
        with open(path, encoding='utf-8') as read:
            for line in read:
                line = line.strip().replace('\\n', '\n')
                if len(line) > 0:
                    cnt = cnt + 1
                    if cnt == 1 and line[0:2] == '1 ':
                        lines_have_ids = True
                    if '\t' in line and (not cands_are_replies):
                        cands_are_replies = True
                        cands = []
                    if lines_have_ids:
                        space_idx = line.find(' ')
                        line = line[space_idx + 1:]
                        if cands_are_replies:
                            sp = line.split('\t')
                            if len(sp) > 1 and sp[1] != '':
                                cands.append(sp[1])
                        else:
                            cands.append(line)
                    else:
                        cands.append(line)
        return cands

    def setup_data(self, path):
        """
        Read data in the fbdialog format.

        Returns ``((x,y,r,c), new_episode?)`` tuples.

        ``x`` represents a query, ``y`` represents the labels, ``r`` represents
        any reward, and ``c`` represents any label_candidates.

        The example above will be translated into the following tuples:

        ::

            x: 'Sam went to the kitchen\\nPat gave Sam the milk\\nWhere is the milk?'
            y: ['kitchen']
            r: '1'
            c: ['hallway', 'kitchen', 'bathroom']
            new_episode = True (this is the first example in the episode)


        ::

            x: 'Sam went to the hallway\\\\nPat went to the bathroom\\\\nWhere is the
                milk?'
            y: ['hallway']
            r: '1'
            c: ['hallway', 'kitchen', 'bathroom']
            new_episode = False (this is the second example in the episode)
        """
        logging.info(f'loading fbdialog data: {path}')
        with open(path, encoding='utf-8') as read:
            start = True
            x = ''
            reward = 0
            last_conv_id = None
            for line in read:
                line = line.strip().replace('\\n', '\n')
                if len(line) == 0:
                    continue
                space_idx = line.find(' ')
                if space_idx == -1:
                    conv_id = int(line)
                else:
                    conv_id = int(line[:space_idx])
                split = line[space_idx + 1:].split('\t')
                for i in range(len(split)):
                    word = split[i].strip()
                    if len(word) == 0:
                        split[i] = ''
                    else:
                        split[i] = word
                if len(split) > 2 and split[2] == '':
                    split[2] = None
                if last_conv_id is None or conv_id <= last_conv_id:
                    x = x.strip()
                    if x:
                        yield ([x, None, reward], start)
                    start = True
                    reward = 0
                    if self.cloze:
                        x = 'Fill in the blank in the last sentence.\n{x}'.format(x=split[0])
                    else:
                        x = split[0]
                elif x:
                    x = '{x}\n{next_x}'.format(x=x, next_x=split[0])
                else:
                    x = split[0]
                last_conv_id = conv_id
                if len(split) > 2 and split[2]:
                    reward += float(split[2])
                if len(split) > 1 and split[1]:
                    split[0] = x
                    split[1] = split[1].split('|')
                    if len(split) > 3:
                        split[3] = split[3].split('|')
                    if len(split) > 2:
                        split[2] = reward
                    else:
                        split.append(reward)
                    if start:
                        yield (split, True)
                        start = False
                    else:
                        yield (split, False)
                    x = ''
                    reward = 0
            if x:
                yield ([x, None, reward], start)