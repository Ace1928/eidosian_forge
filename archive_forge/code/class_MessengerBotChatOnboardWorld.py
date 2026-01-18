from parlai.core.worlds import World
from parlai.chat_service.services.messenger.worlds import OnboardWorld
from parlai.core.agents import create_agent_from_shared
class MessengerBotChatOnboardWorld(OnboardWorld):
    """
    Example messenger onboarding world for Chatbot Model.
    """

    @staticmethod
    def generate_world(opt, agents):
        return MessengerBotChatOnboardWorld(opt=opt, agent=agents[0])

    def parley(self):
        self.episodeDone = True