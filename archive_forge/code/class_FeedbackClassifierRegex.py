import re
import torch
class FeedbackClassifierRegex(object):

    def __init__(self):
        self.failure_regexes = {ISAID: 'i .*(?:said|asked|told).*', NOTSENSE: "((not|nt|n't).*mak.*sense)|(mak.*no .*sense)", UM: 'u(m|h)+\\W', YOUWHAT: 'you.*what\\?', WHATYOU: 'what.*you (?:mean|refer|talk).*\\?', WHATDO: 'what.*to do with.*\\?'}

    def predict_proba(self, contexts):
        probs = []
        for context in contexts:
            start = context.rindex('__p1__')
            try:
                end = context.index('__null__')
            except ValueError:
                end = len(context)
            last_response = context[start:end]
            failure_mode = self.identify_failure_mode(last_response)
            probs.append(failure_mode is None)
        return torch.FloatTensor(probs)

    def identify_failure_mode(self, text):
        if re.search(self.failure_regexes[ISAID], text, flags=re.I):
            return ISAID
        elif re.search(self.failure_regexes[NOTSENSE], text, flags=re.I):
            return NOTSENSE
        elif re.search(self.failure_regexes[UM], text, flags=re.I):
            return UM
        elif re.search(self.failure_regexes[YOUWHAT], text, flags=re.I):
            return YOUWHAT
        elif re.search(self.failure_regexes[WHATYOU], text, flags=re.I):
            return WHATYOU
        elif re.search(self.failure_regexes[WHATDO], text, flags=re.I):
            return WHATDO
        else:
            return None