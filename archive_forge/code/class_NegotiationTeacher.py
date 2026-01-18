from parlai.core.teachers import Teacher
from .build import build
import os
import random
class NegotiationTeacher(Teacher):
    """
    End-to-end negotiation teacher that loads the data from
    https://github.com/facebookresearch/end-to-end-negotiator.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.datatype = opt['datatype'].split(':')[0]
        self.datatype_ = opt['datatype']
        self.random = self.datatype_ == 'train'
        build(opt)
        filename = 'val' if self.datatype == 'valid' else self.datatype
        data_path = os.path.join(opt['datapath'], 'negotiation', 'end-to-end-negotiator-bbb93bbf00f69fced75d5c0d22e855bda07c9b78', 'src', 'data', 'negotiate', filename + '.txt')
        if shared and 'data' in shared:
            self.episodes = shared['episodes']
        else:
            self._setup_data(data_path)
        self.step_size = opt.get('batchsize', 1)
        self.data_offset = opt.get('batchindex', 0)
        self.reset()

    def num_examples(self):
        num_exs = 0
        dialogues = [self._split_dialogue(get_tag(episode.strip().split(), DIALOGUE_TAG)) for episode in self.episodes]
        num_exs = sum((len([d for d in dialogue if YOU_TOKEN in d]) + 1 for dialogue in dialogues))
        return num_exs

    def num_episodes(self):
        return len(self.episodes)

    def reset(self):
        super().reset()
        self.episode_idx = self.data_offset - self.step_size
        self.dialogue_idx = None
        self.expected_reponse = None
        self.epochDone = False

    def share(self):
        shared = super().share()
        shared['episodes'] = self.episodes
        return shared

    def _setup_data(self, data_path):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.episodes = data_file.readlines()

    def observe(self, observation):
        """
        Process observation for metrics.
        """
        if self.expected_reponse is not None:
            self.metrics.evaluate_response(observation, self.expected_reponse)
            self.expected_reponse = None
        return observation

    def act(self):
        if self.dialogue_idx is not None:
            return self._continue_dialogue()
        elif self.random:
            self.episode_idx = random.randrange(len(self.episodes))
            return self._start_dialogue()
        elif self.episode_idx + self.step_size >= len(self.episodes):
            self.epochDone = True
            return {'episode_done': True}
        else:
            self.episode_idx = (self.episode_idx + self.step_size) % len(self.episodes)
            return self._start_dialogue()

    def _split_dialogue(self, words, separator=EOS_TOKEN):
        sentences = []
        start = 0
        for stop in range(len(words)):
            if words[stop] == separator:
                sentences.append(words[start:stop])
                start = stop + 1
        if stop >= start:
            sentences.append(words[start:])
        return sentences

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

    def _continue_dialogue(self, skip_teacher=False):
        action = {}
        if not skip_teacher:
            self.dialogue_idx += 1
            if self.dialogue_idx >= len(self.dialogue):
                action['text'] = SELECTION_TOKEN
            else:
                sentence = self.dialogue[self.dialogue_idx]
                assert sentence[0] == THEM_TOKEN
                action['text'] = ' '.join(sentence[1:])
        self.dialogue_idx += 1
        if self.datatype.startswith('train'):
            if self.dialogue_idx >= len(self.dialogue):
                self.expected_reponse = [' '.join(self.output)]
            else:
                sentence = self.dialogue[self.dialogue_idx]
                assert sentence[0] == YOU_TOKEN
                self.expected_reponse = [' '.join(sentence[1:])]
            action['labels'] = self.expected_reponse
        if self.dialogue_idx >= len(self.dialogue):
            self.dialogue_idx = None
            action['episode_done'] = True
        else:
            action['episode_done'] = False
        return action