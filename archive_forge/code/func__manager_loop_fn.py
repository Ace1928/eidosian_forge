import json
import asyncio
import logging
from parlai.core.agents import create_agent
from parlai.chat_service.core.chat_service_manager import ChatServiceManager
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.misc as utils
from parlai.chat_service.services.websocket.sockets import MessageSocketHandler
from agents import WebsocketAgent
import tornado
from tornado.options import options
def _manager_loop_fn(self):
    """
        An iteration of the manager's main loop to launch worlds.
        """
    with self.agent_pool_change_condition:
        valid_pools = self._get_unique_pool()
        for world_type, agent_pool in valid_pools.items():
            world_config = self.task_configs[world_type]
            if world_config.max_time_in_pool is not None:
                self.check_timeout_in_pool(world_type, agent_pool, world_config.max_time_in_pool, world_config.backup_task)
            needed_agents = self.max_agents_for[world_type]
            if len(agent_pool) >= needed_agents:
                log_utils.print_and_log(logging.INFO, 'starting pool', should_print=True)
                self.conversation_index += 1
                task_id = 't_{}'.format(self.conversation_index)
                agent_states = [w for w in agent_pool[:needed_agents]]
                agents = []
                for state in agent_states:
                    agent = self._create_agent(task_id, state.get_id())
                    agent.onboard_data = state.onboard_data
                    agent.data = state.data
                    state.assign_agent_to_task(agent, task_id)
                    state.set_active_agent(agent)
                    agents.append(agent)
                    state.stored_data['seen_wait_message'] = False
                assign_role_function = utils.get_assign_roles_fn(self.world_module, self.taskworld_map[world_type])
                if assign_role_function is None:
                    assign_role_function = utils.default_assign_roles_fn
                assign_role_function(agents)
                for a in agents:
                    self.remove_agent_from_pool(self.get_agent_state(a.id), world_type=world_type, mark_removed=False)
                for a in agents:
                    partner_list = agents.copy()
                    partner_list.remove(a)
                    a.message_partners = partner_list
                done_callback = self._get_done_callback_for_agents(task_id, world_type, agents)
                future = self.world_runner.launch_task_world(task_id, self.taskworld_map[world_type], agents)
                future.add_done_callback(done_callback)
                self.active_worlds[task_id] = future