from parlai.core.worlds import create_task
from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld
from parlai.tasks.interactive.worlds import InteractiveWorld as InteractiveBaseWorld
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
import random
import pickle
import os
class InteractiveSimpleWorld(InteractiveBaseWorld):

    @staticmethod
    def add_cmdline_args(argparser):
        parser = argparser.add_argument_group('LIGHT Interactive World')
        parser.add_argument('--add-task-string', type='bool', default=False, help='Add _task_speech to text input to model or not')

    def init_contexts(self, shared=None):
        light_opt = self.opt.copy()
        light_opt['task'] = 'light_dialog'
        light_opt['interactive_task'] = False
        light_agent = RepeatLabelAgent(light_opt)
        self.light_world = create_task(light_opt, light_agent)
        self.cnt = 0

    def get_contexts(self):
        while True:
            self.light_world.parley()
            msg = self.light_world.get_acts()[0]
            if msg.get('episode_done', False):
                self.light_world.parley()
                msg = self.light_world.get_acts()[0]
                break
        txt = msg.get('text', '').split('\n')
        a1_persona = ''
        a2_persona = ''
        p = {}
        for t in txt:
            p[t.split(' ')[0]] = t
        if self.opt['add_task_string']:
            task_name = ' _task_speech\n'
        else:
            task_name = ''
        a1_persona = task_name + p['_setting_name'] + '\n' + p['_setting_desc'] + '\n' + p['_self_name'].replace('_self_name', '_partner_name') + '\n' + p['_partner_name'].replace('_partner_name', '_self_name') + '\n' + '_self_persona I am a ' + ' '.join(p['_partner_name'].split(' ')[1:]) + '.'
        a2_persona = task_name + p['_setting_name'] + '\n' + p['_setting_desc'] + '\n' + p['_partner_name'] + '\n' + p['_self_name'] + '\n' + p['_self_persona']
        return (a1_persona, a2_persona)