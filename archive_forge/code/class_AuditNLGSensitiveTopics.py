import os
import torch
import warnings
from typing import List
from string import Template
from enum import Enum
class AuditNLGSensitiveTopics(object):

    def __init__(self):
        pass

    def __call__(self, output_text, **kwargs):
        try:
            from auditnlg.safety.exam import safety_scores
        except ImportError as e:
            print('Could not import optional dependency: auditnlg\nPlease install manually with:\n pip install auditnlg\nFollowed by:\npip install -r requirements.txt')
            raise e
        data = [{'output': output_text}]
        result = safety_scores(data=data, method='sensitive_topics')
        scores = result[1]['all_scores'][0]
        is_safe = scores['pred_class'] == 'none'
        report = ''
        if not is_safe:
            report += f'Predicted class: {scores['pred_class']}\n'
            report += '|' + '|'.join((f'{n:^10}' for n in [list(k.keys())[0] for k in scores['class_scores']])) + '|\n'
            report += '|' + '|'.join((f'{n:^10.5}' for n in [list(k.values())[0] for k in scores['class_scores']])) + '|\n'
        return ('Sensitive Topics', is_safe, report)