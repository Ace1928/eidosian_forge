import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
class AssignState:
    """
    Class for holding state information about an assignment currently claimed by an
    agent.
    """
    STATUS_NONE = 'none'
    STATUS_ONBOARDING = 'onboarding'
    STATUS_WAITING = 'waiting'
    STATUS_IN_TASK = 'in task'
    STATUS_DONE = 'done'
    STATUS_DISCONNECT = 'disconnect'
    STATUS_PARTNER_DISCONNECT = 'partner disconnect'
    STATUS_PARTNER_DISCONNECT_EARLY = 'partner disconnect early'
    STATUS_EXPIRED = 'expired'
    STATUS_RETURNED = 'returned'
    STATUS_STATIC = 'static'

    def __init__(self, status=None):
        """
        Create an AssignState to track the state of an agent's assignment.
        """
        if status is None:
            status = self.STATUS_NONE
        self.status = status
        self.messages = []
        self.message_ids = []

    def clear_messages(self):
        self.messages = []
        self.message_ids = []

    def append_message(self, message):
        """
        Appends a message to the list of messages, ensures that it is not a duplicate
        message.
        """
        if message['message_id'] in self.message_ids:
            return
        self.message_ids.append(message['message_id'])
        self.messages.append(message)

    def get_messages(self):
        return self.messages

    def set_status(self, status):
        """
        Set the status of this agent on the task.
        """
        self.status = status

    def get_status(self):
        """
        Get the status of this agent on its task.
        """
        return self.status

    def is_final(self):
        """
        Return True if the assignment is in a final status that can no longer be acted
        on.
        """
        return self.status == self.STATUS_DISCONNECT or self.status == self.STATUS_DONE or self.status == self.STATUS_PARTNER_DISCONNECT or (self.status == self.STATUS_PARTNER_DISCONNECT_EARLY) or (self.status == self.STATUS_RETURNED) or (self.status == self.STATUS_EXPIRED)