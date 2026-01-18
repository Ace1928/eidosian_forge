from parlai.core.worlds import World
from parlai.chat_service.services.messenger.worlds import OnboardWorld
class MessengerOverworld(World):
    """
    World to handle moving agents to their proper places.
    """
    DEMOS = {'echo': (MessengerEchoOnboardWorld, MessengerEchoTaskWorld), 'onboard data': (MessengerOnboardDataOnboardWorld, MessengerOnboardDataTaskWorld), 'chat': (MessengerChatOnboardWorld, MessengerChatTaskWorld), 'EXIT': (None, None)}

    def __init__(self, opt, agent):
        self.agent = agent
        self.opt = opt
        self.first_time = True
        self.episodeDone = False

    @staticmethod
    def generate_world(opt, agents):
        return MessengerOverworld(opt, agents[0])

    @staticmethod
    def assign_roles(agents):
        for a in agents:
            a.disp_id = 'Agent'

    def episode_done(self):
        return self.episodeDone

    def parley(self):
        if self.first_time:
            self.agent.observe({'id': 'Overworld', 'text': 'Welcome to the overworld for the ParlAI messenger demo. Choose one of the demos from the listed quick replies. ', 'quick_replies': self.DEMOS.keys()})
            self.first_time = False
        a = self.agent.act()
        if a is not None:
            if a['text'] in self.DEMOS:
                self.agent.observe({'id': 'Overworld', 'text': 'Transferring to ' + a['text']})
                self.episodeDone = True
                return a['text']
            else:
                self.agent.observe({'id': 'Overworld', 'text': 'Invalid option. Choose one of the demos from the listed quick replies. ', 'quick_replies': self.DEMOS.keys()})