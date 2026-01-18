from parlai.core.worlds import World
from parlai.chat_service.services.messenger.worlds import OnboardWorld
class MessengerChatOnboardWorld(OnboardWorld):
    """
    Example messenger onboarding world for chat task, displays intro and explains
    instructions.
    """

    def __init__(self, opt, agent):
        self.agent = agent
        self.episodeDone = False
        self.turn = 0
        self.data = {}

    @staticmethod
    def generate_world(opt, agents):
        return MessengerChatOnboardWorld(opt, agents[0])

    @staticmethod
    def assign_roles(agents):
        for a in agents:
            a.disp_id = 'Agent'

    def parley(self):
        if self.turn == 0:
            self.agent.observe({'id': 'Onboarding', 'text': 'Welcome to the onboarding world free chat. Enter your display name.'})
            a = self.agent.act()
            while a is None:
                a = self.agent.act()
            self.data['user_name'] = a['text']
            self.turn = self.turn + 1
        elif self.turn == 1:
            self.agent.observe({'id': 'Onboarding', 'text': 'You will be matched with a random person. Say [DONE] to end the chat.'})
            self.episodeDone = True