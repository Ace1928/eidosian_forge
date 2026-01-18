import logging
import math
import os
import pickle
import threading
import time
import uuid
import errno
import requests
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.worker_manager import WorkerManager
from parlai.mturk.core.dev.mturk_data_handler import MTurkDataHandler
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.mturk_utils as mturk_utils
import parlai.mturk.core.dev.server_utils as server_utils
import parlai.mturk.core.dev.shared_utils as shared_utils
def _task_function(opt, agents, conversation_id):
    """
            Wait for agents to join the world, then run task function.
            """
    shared_utils.print_and_log(logging.INFO, 'Starting task {}...'.format(conversation_id))
    shared_utils.print_and_log(logging.DEBUG, 'Waiting for all agents to join the conversation...')
    start_time = time.time()
    while True:
        all_joined = True
        for agent in agents:
            if agent.get_status() != AssignState.STATUS_IN_TASK:
                all_joined = False
        if all_joined:
            break
        if time.time() - start_time > WORLD_START_TIMEOUT:
            shared_utils.print_and_log(logging.INFO, 'Timeout waiting for {}, move back to waiting'.format(conversation_id))
            self._move_agents_to_waiting(agents)
            return
        time.sleep(shared_utils.THREAD_SHORT_SLEEP)
    shared_utils.print_and_log(logging.INFO, 'All agents joined the conversation {}!'.format(conversation_id))
    self.started_conversations += 1
    world = get_task_world(mturk_manager=self, opt=opt, workers=agents)
    try:
        while not world.episode_done():
            world.parley()
    except AgentTimeoutError as e:
        self.handle_turker_timeout(e.worker_id, e.assignment_id)
    except AbsentAgentError:
        pass
    world.shutdown()
    world.review_work()
    save_data = world.prep_save_data(agents)
    if save_data is not None:
        MTurkDataHandler.save_world_data(save_data, self.task_group_id, conversation_id, sandbox=self.is_sandbox)
    for agent in agents:
        agent.clear_messages()
    if self._no_agents_incomplete(agents):
        self.completed_conversations += 1
    if self.opt['max_connections'] > 0:
        if self.accepting_workers:
            for agent in agents:
                if agent.submitted_hit():
                    self.create_additional_hits(1)
                    self.hit_id_list.remove(agent.hit_id)