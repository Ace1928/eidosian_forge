import json
import random
from parlai.tasks.blended_skill_talk.agents import raw_data_path, safe_personas_path
from parlai.tasks.interactive.worlds import InteractiveWorld as InteractiveBaseWorld
from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld
class InteractiveWorld(InteractiveBaseWorld):

    @staticmethod
    def add_cmdline_args(argparser):
        parser = argparser.add_argument_group('BST Interactive World')
        parser.add_argument('--display-partner-persona', type='bool', default=True, help='Display your partner persona at the end of the chat')
        parser.add_argument('--include-personas', type='bool', default=True, help='Include personas as input context, or not')
        parser.add_argument('--include-initial-utterances', type='bool', default=False, help='Include context conversation at beginning or not')
        parser.add_argument('--safe-personas-only', type='bool', default=True, help='Only use personas on an allowed list of safe personas', hidden=True)

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        self.display_partner_persona = self.opt['display_partner_persona']

    def init_contexts(self, shared=None):
        self.contexts_data = get_contexts_data(self.opt, shared=shared)

    def get_contexts(self):
        random.seed()
        p = random.choice(self.contexts_data)
        return (p[0], p[1])

    def finalize_episode(self):
        print('\nCHAT DONE.\n')
        if self.display_partner_persona:
            partner_persona = self.p2.replace('your persona:', "partner's persona:")
            print(f'Your partner was playing the following persona:\n{partner_persona}')
        if not self.epoch_done():
            print('\n[ Preparing new chat ... ]\n')

    def share(self):
        shared_data = super().share()
        shared_data['contexts_data'] = self.contexts_data
        return shared_data