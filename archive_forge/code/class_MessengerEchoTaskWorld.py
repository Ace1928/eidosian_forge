from parlai.core.worlds import World
from parlai.chat_service.services.messenger.worlds import OnboardWorld
class MessengerEchoTaskWorld(World):
    """
    Example one person world that uses only user input.
    """
    MAX_AGENTS = 1

    def __init__(self, opt, agent):
        self.agent = agent
        self.episodeDone = False

    @staticmethod
    def generate_world(opt, agents):
        return MessengerEchoTaskWorld(opt, agents[0])

    @staticmethod
    def assign_roles(agents):
        agents[0].disp_id = 'EchoAgent'

    def parley(self):
        a = self.agent.act()
        if a is not None:
            if '[DONE]' in a['text']:
                self.episodeDone = True
            else:
                a['id'] = 'World'
                self.agent.observe(a)

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        self.agent.shutdown()