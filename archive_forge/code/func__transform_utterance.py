from parlai.core.teachers import DialogTeacher
@staticmethod
def _transform_utterance(utterance, user_types):
    uid = utterance['userId']
    t = user_types[uid]
    return ': '.join([utterance['userId'] + '(' + t + ')', utterance['text']])