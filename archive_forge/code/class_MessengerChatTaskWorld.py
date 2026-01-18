from parlai.core.worlds import World
from parlai.chat_service.services.messenger.worlds import OnboardWorld
class MessengerChatTaskWorld(World):
    """
    Example one person world that lets two users chat.
    """
    MAX_AGENTS = 2

    def __init__(self, opt, agents):
        self.agents = agents
        self.episodeDone = False

    @staticmethod
    def generate_world(opt, agents):
        return MessengerChatTaskWorld(opt, agents)

    @staticmethod
    def assign_roles(agents):
        for a in agents:
            a.disp_id = 'Agent'

    def parley(self):
        for x in [0, 1]:
            a = self.agents[x].act()
            if a is not None:
                if '[DONE]' in a['text']:
                    self.agents[x - 1].observe({'id': 'World', 'text': 'The other agent has ended the chat.'})
                    self.episodeDone = True
                else:
                    self.agents[x - 1].observe(a)

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        for agent in self.agents:
            agent.shutdown()