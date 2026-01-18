from parlai.core.teachers import Teacher
from .build import build
import os
import random
def _start_dialogue(self):
    words = self.episodes[self.episode_idx].strip().split()
    self.values = get_tag(words, INPUT_TAG)
    self.dialogue = self._split_dialogue(get_tag(words, DIALOGUE_TAG))
    self.output = get_tag(words, OUTPUT_TAG)
    assert self.dialogue[-1][1] == SELECTION_TOKEN
    book_cnt, book_val, hat_cnt, hat_val, ball_cnt, ball_val = self.values
    welcome = WELCOME_MESSAGE.format(book_cnt=book_cnt, book_val=book_val, hat_cnt=hat_cnt, hat_val=hat_val, ball_cnt=ball_cnt, ball_val=ball_val)
    self.dialogue_idx = -1
    if self.dialogue[0][0] == THEM_TOKEN:
        action = self._continue_dialogue()
        action['text'] = welcome + '\n' + action['text']
    else:
        action = self._continue_dialogue(skip_teacher=True)
        action['text'] = welcome
    action['items'] = {'book_cnt': book_cnt, 'book_val': book_val, 'hat_cnt': hat_cnt, 'hat_val': hat_val, 'ball_cnt': ball_cnt, 'ball_val': ball_val}
    return action