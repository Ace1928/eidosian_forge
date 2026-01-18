from parlai.core.worlds import World
from parlai.chat_service.services.messenger.worlds import OnboardWorld
class MessengerEchoOnboardWorld(OnboardWorld):
    """
    Example messenger onboarding world for Echo task, displays.

    onboarding worlds that only exist to send an introduction message.
    """

    @staticmethod
    def generate_world(opt, agents):
        return MessengerEchoOnboardWorld(opt=opt, agent=agents[0])

    def parley(self):
        self.agent.observe({'id': 'Onboarding', 'text': 'Welcome to the onboarding world for our echo bot. The next message you send will be echoed. Use [DONE] to finish the chat.'})
        self.episodeDone = True