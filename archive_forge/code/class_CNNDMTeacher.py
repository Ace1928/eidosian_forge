from parlai.core.teachers import DialogTeacher
from .build import build
import os
import unicodedata
class CNNDMTeacher(DialogTeacher):

    def __init__(self, opt, shared=None):
        self.dt = opt.get('datatype', 'train').split(':')[0]
        self.id = 'cnn_dm'
        self.datapath = os.path.join(opt['datapath'], 'CNN_DM')
        opt['datafile'] = self._path(opt)
        super().__init__(opt, shared)

    def _path(self, opt):
        build(opt)
        dt = opt['datatype'].split(':')[0]
        return os.path.join(self.datapath, dt + '.txt')

    def setup_data(self, input_path):
        self.question = 'What is the summary?'
        new_episode = True
        num_missing = 0
        num_added = 0
        print('loading: ' + input_path)
        with open(input_path) as stories_file:
            for story in stories_file:
                try:
                    story_file = open(os.path.join(self.datapath, story.strip()))
                except EnvironmentError:
                    num_missing += 1
                    continue
                num_added += 1
                article, highlights = ([], [])
                is_highlight = False
                for line in story_file:
                    line = _fix_missing_period(line.strip())
                    if line == '':
                        continue
                    if line.startswith('@highlight'):
                        is_highlight = True
                        continue
                    if is_highlight:
                        highlights.append(line)
                    else:
                        article.append(line)
                text = unicodedata.normalize('NFKC', ' '.join(article)) + '\n' + self.question
                label = [unicodedata.normalize('NFKC', ' '.join(highlights))]
                yield ((text, label, None, None), new_episode)
        print('{} stories added, {} stories missing.'.format(num_added, num_missing))