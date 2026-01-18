from parlai.core.teachers import DialogTeacher, ChunkTeacher, ChunkOutput
from parlai.core.message import Message
from .build import build
import json
import os
from typing import List, Tuple
class SummaryTeacher(DialogTeacher):
    """
    Reads Wikipedia pages one at a time, only uses summaries.
    """

    def __init__(self, opt, shared=None):
        self.key_value = ':key-value' in opt['task']
        opt['task'] = 'wikipedia:summary'
        build(opt)
        opt['datafile'] = os.path.join(opt['datapath'], 'wikipedia/summary/summaries.json')
        self.id = 'wikipedia'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path) as wf:
            for article_json in wf:
                article = json.loads(article_json)
                title = article['title']
                text = article['text']
                if self.key_value:
                    yield ((title, [text]), True)
                else:
                    yield ((title + '\n' + text, ['']), True)