from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.core.worlds import validate
from parlai.mturk.tasks.personachat.personachat_chat.extract_and_save_personas import (
from joblib import Parallel, delayed
import numpy as np
import time
import os
import pickle
import random
class RephrasePersonaWorld(MTurkOnboardWorld):
    """
    A world that provides a persona to the MTurkAgent.
    """

    def __init__(self, opt, mturk_agent):
        self.opt = opt
        self.personas_generator = opt['personas_generator']
        self.persona_idx, self.persona = self.personas_generator.pop_persona()
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.episodeDone = False
        self.rephrased_persona = []
        self.shown_persona = False
        self.max_response_time = opt['max_resp_time']
        self.mturk_agent = mturk_agent
        self.num_done = 0
        self.completed = False
        super().__init__(opt, mturk_agent)

    def parley(self):
        if not self.shown_persona:
            self.mturk_agent.observe({'id': 'SYSTEM', 'text': 'You will be asked to rephrase the following persona:\n\n ' + '\n'.join(self.persona)})
            self.shown_persona = True
        if len(self.rephrased_persona) < len(self.persona):
            persona_done = False
            self.mturk_agent.observe({'id': 'SYSTEM', 'text': "Please rephrase the sentence below so that it sticks to the same person's characteristics: \n\n<b><span style='color:blue'>{}</span></b>\n\n There are {} sentences left to be rephrased.".format(self.persona[self.num_done], len(self.persona) - len(self.rephrased_persona) - 1)})
            while not persona_done:
                act = self.mturk_agent.act(timeout=self.max_response_time)
                if act['episode_done']:
                    self.episodeDone = True
                    return
                if self.is_msg_tooshortlong(act, self.mturk_agent) or self.is_close_match(act, self.mturk_agent, self.persona[self.num_done]):
                    pass
                else:
                    self.rephrased_persona.append(act['text'])
                    self.num_done += 1
                    persona_done = True
            return
        else:
            self.mturk_agent.observe({'id': 'SYSTEM', 'text': 'Thank you for rephrasing the character! \n <b><span style="color:blue">Please click "Done with this HIT" below to submit the HIT.</span></b>'})
            self.episodeDone = True
            return

    def save_data(self):
        data_path = self.opt['extract_personas_path'] + '/rephrased_personas'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if len(self.rephrased_persona) == len(self.persona):
            self.completed = True
            self.personas_generator.add_done_persona(self.persona_idx)
            filename = os.path.join(data_path, 'persona_{}_{}_{}.pkl'.format(time.strftime('%Y%m%d-%H%M%S'), self.mturk_agent.worker_id, self.task_type))
            print('Profile successfully saved at {}.'.format(filename))
            pickle.dump({'hit_id': self.mturk_agent.hit_id, 'assignment_id': self.mturk_agent.assignment_id, 'worker_id': self.mturk_agent.worker_id, 'persona': self.persona, 'rephrased': self.rephrased_persona, 'persona_idx': self.persona_idx}, open(filename, 'wb'))
        else:
            self.personas_generator.push_persona(self.persona_idx)
            print('Incomplete persona:', self.persona_idx)
            filename = os.path.join(data_path, 'incomplete_persona_{}_{}_{}.pkl'.format(time.strftime('%Y%m%d-%H%M%S'), self.mturk_agent.worker_id, self.task_type))
            print('Profile successfully saved at {}.'.format(filename))
            pickle.dump({'hit_id': self.mturk_agent.hit_id, 'assignment_id': self.mturk_agent.assignment_id, 'worker_id': self.mturk_agent.worker_id, 'persona': self.persona, 'rephrased': self.rephrased_persona, 'persona_idx': self.persona_idx}, open(filename, 'wb'))

    def shutdown(self):
        global shutdown_agent

        def shutdown_agent(mturk_agent):
            mturk_agent.shutdown()
        Parallel(n_jobs=1, backend='threading')((delayed(shutdown_agent)(agent) for agent in [self.mturk_agent]))

    def is_exact_match(self, act, ag, persona_data, tolerance=0):
        if act['episode_done']:
            return False
        control_msg = {'episode_done': False}
        control_msg['id'] = 'SYSTEM'
        text = act['text']
        if text not in ['', ' ', '  ', '   ']:
            n_word_match = 0
            for per in [persona_data]:
                per_parse = per.split(' ')
                regular_words = ['', ' ', 'I', "I'm", 'My', 'i']
                for r_w in regular_words:
                    if r_w in per_parse:
                        per_parse.remove(r_w)
                per_subseq = [' '.join(per_parse[i:i + len(per_parse) - tolerance]) for i in range(tolerance + 1)]
                for pp in per_subseq:
                    if pp in ['', ' ', '  ', '   ']:
                        per_subseq.remove(pp)
                n_word_match += sum([paa in text for paa in per_subseq])
            if n_word_match > 0:
                control_msg['text'] = 'We found that you <b><span style="color:red">trivially copied character descriptions</span></b>. Please rephrase your message again.'
                ag.observe(validate(control_msg))
                return True
            else:
                return False

    def is_close_match(self, act, ag, persona_data, tolerance=0.7):
        if act['episode_done']:
            return False
        control_msg = {'episode_done': False}
        control_msg['id'] = 'SYSTEM'
        text = act['text']
        if text not in ['', ' ', '  ', '   ']:
            n_word_match = 0
            per_parse = persona_data.split(' ')
            regular_words = ['', ' ', 'I', "I'm", 'My', 'i']
            for r_w in regular_words:
                if r_w in per_parse:
                    per_parse.remove(r_w)
            n_word_match += sum([word in text for word in per_parse])
            if n_word_match / (len(per_parse) + 1) > tolerance:
                control_msg['text'] = 'We found that you <b><span style="color:red">trivially copied character descriptions</span></b>. Please rephrase your message again.'
                ag.observe(validate(control_msg))
                return True
            else:
                return False

    def is_msg_tooshortlong(self, act, ag, th_min=3, th_max=17):
        if act['episode_done']:
            return False
        control_msg = {'episode_done': False}
        control_msg['id'] = 'SYSTEM'
        msg_len = len(act['text'].split(' '))
        if msg_len < th_min:
            control_msg['text'] = 'Your message is too short, please make it more than <b><span style="color:red">5 words</span></b>.'
            ag.observe(validate(control_msg))
            return True
        if msg_len > th_max:
            control_msg['text'] = 'Your message is too long, please make it less than <b><span style="color:red">15 words</span></b>.'
            ag.observe(validate(control_msg))
            return True
        return False

    def review_work(self):
        if self.completed:
            self.mturk_agent.approve_work()

    def episode_done(self):
        return self.episodeDone