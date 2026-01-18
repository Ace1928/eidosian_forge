import parlai.core.build_data as build_data
import os
import json
from parlai.core.build_data import DownloadableFile
def _handle_paragraph(each):
    output = []
    story = each['context'].replace('\n', '\\n')
    for idx, q_a in enumerate(each['qas']):
        question_txt = ''
        if idx == 0:
            question_txt = story + '\\n' + q_a['question']
        else:
            question_txt = q_a['question']
        starts, labels = _parse_answers(q_a)
        output.append(OUTPUT_FORMAT.format(question=question_txt, continuation=MAP_CONTINUATION.get(q_a['followup']), affirmation=MAP_AFFIRMATION.get(q_a['yesno']), start=starts, labels=labels))
        if idx < len(each['qas']) - 1:
            output.append('\n')
    output.append('\t\tepisode_done:True\n')
    return ''.join(output)