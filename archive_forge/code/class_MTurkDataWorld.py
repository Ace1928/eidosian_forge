from parlai.core.worlds import World
from parlai.mturk.core.dev.agents import AssignState
class MTurkDataWorld(World):

    def prep_save_data(self, workers):
        """
        This prepares data to be saved for later review, including chats from individual
        worker perspectives.
        """
        custom_data = self.get_custom_task_data()
        save_data = {'custom_data': custom_data, 'worker_data': {}}
        for agent in workers:
            messages = agent.get_messages()
            save_messages = [m for m in messages if m.get('text') != '[PEER_REVIEW]']
            save_data['worker_data'][agent.worker_id] = {'worker_id': agent.worker_id, 'agent_id': agent.id, 'assignment_id': agent.assignment_id, 'messages': save_messages, 'given_feedback': agent.feedback, 'completed': self.episode_done()}
        if len(workers) == 2:
            data = save_data['worker_data']
            a_0 = workers[0]
            a_1 = workers[1]
            data[a_0.worker_id]['received_feedback'] = a_1.feedback
            data[a_1.worker_id]['received_feedback'] = a_0.feedback
        return save_data

    def get_custom_task_data(self):
        """
        This function should take the contents of whatever was collected during this
        task that should be saved and return it in some format, preferrably a dict
        containing acts.

        If data needs pickling, put it in a field named 'needs-pickle'
        """
        pass