import os
import torch
import warnings
from typing import List
from string import Template
from enum import Enum
class SalesforceSafetyChecker(object):

    def __init__(self):
        pass

    def __call__(self, output_text, **kwargs):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
        config = AutoConfig.from_pretrained('Salesforce/safety-flan-t5-base')
        tokenizer = AutoTokenizer.from_pretrained('Salesforce/safety-flan-t5-base')
        model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/safety-flan-t5-base', config=config)
        model.eval()
        prefix = 'Is the <Text> field safe or unsafe?'
        input_ids = tokenizer(prefix + ' <Text> ' + output_text + ' <Context> ', return_tensors='pt').input_ids
        if len(input_ids[0]) > 512:
            warnings.warn('Input length is > 512 token. Safety check result could be incorrect.')
        with torch.no_grad():
            outputs = model.generate(input_ids, output_scores=True, return_dict_in_generate=True, max_new_tokens=20)
        is_safe = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).split(' ')[0] == 'safe'
        report = ''
        if not is_safe:
            true_false_ids = tokenizer('true false').input_ids[:2]
            keys = ['toxicity', 'hate', 'identity', 'violence', 'physical', 'sexual', 'profanity', 'biased']
            scores = {}
            for k, i in zip(keys, range(3, 20, 2)):
                scores[k] = round(outputs.scores[i][0, true_false_ids].softmax(dim=0)[0].item(), 5)
            report += '|' + '|'.join((f'{n:^10}' for n in scores.keys())) + '|\n'
            report += '|' + '|'.join((f'{n:^10}' for n in scores.values())) + '|\n'
        return ('Salesforce Content Safety Flan T5 Base', is_safe, report)

    def get_total_length(self, data):
        prefix = 'Is the <Text> field safe or unsafe '
        input_sample = '<Text> {output} <Context> '.format(**data[0])
        return len(self.tokenizer(prefix + input_sample)['input_ids'])