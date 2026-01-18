from parlai.core.worlds import World
from parlai.chat_service.services.messenger.worlds import OnboardWorld
class MessengerOnboardDataTaskWorld(World):
    """
    Example one person world that relays data given in the onboard world.
    """
    MAX_AGENTS = 1

    def __init__(self, opt, agent):
        self.agent = agent
        self.episodeDone = False

    @staticmethod
    def generate_world(opt, agents):
        return MessengerOnboardDataTaskWorld(opt, agents[0])

    @staticmethod
    def assign_roles(agents):
        agents[0].disp_id = 'DataAgent'

    def parley(self):
        name = self.agent.onboard_data['name']
        color = self.agent.onboard_data['color']
        self.agent.observe({'id': 'World', 'text': 'During onboarding, you said your name was {} and your favorite color was {}'.format(name, color)})
        self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        self.agent.shutdown()