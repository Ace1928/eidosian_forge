from concurrent import futures
from collections import namedtuple
import copy
import logging
import sys
import threading
from s3transfer.compat import MAXINT
from s3transfer.compat import six
from s3transfer.exceptions import CancelledError, TransferNotDoneError
from s3transfer.utils import FunctionContainer
from s3transfer.utils import TaskSemaphore
class BaseTransferMeta(object):

    @property
    def call_args(self):
        """The call args used in the transfer request"""
        raise NotImplementedError('call_args')

    @property
    def transfer_id(self):
        """The unique id of the transfer"""
        raise NotImplementedError('transfer_id')

    @property
    def user_context(self):
        """A dictionary that requesters can store data in"""
        raise NotImplementedError('user_context')