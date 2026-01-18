from parlai.core.torch_agent import TorchAgent
from .controls import eval_attr
class ConvAI2History(object):
    """
    An object to store the history of a ConvAI2 conversation in a convenient format.
    """

    def __init__(self, text, assume_persontokens=True, dictionary=None):
        """
        Separates text (the dialogue context) into:
          self.persona_lines: list of strings; the persona lines with "your persona:"
            removed.
          self.partner_utts: list of strings; the partner's utterances.
          self.own_utts: list of strings; the bot's own utterances.
        All of the above are lowercase and tokenized.

        Inputs:
          text: string. Contains several lines separated by 
. The first few lines are
            persona lines beginning "your persona: ", then the next lines are dialogue.
          assume_persontokens: If True, assert that the dialogue lines start with
            __p1__ and __p2__ respectively, and then remove them.
          dictionary: parlai DictionaryAgent.
        """
        p1_token, p2_token = (TorchAgent.P1_TOKEN, TorchAgent.P2_TOKEN)
        lines = [t.strip() for t in text.split('\n')]
        persona_lines = [t for t in lines if 'your persona:' in t]
        persona_lines = [remove_prefix(pl, 'your persona:') for pl in persona_lines]
        persona_lines = [fix_personaline_period(pl) for pl in persona_lines]
        utts = lines[len(persona_lines):]
        p1_utts = [utts[i] for i in range(0, len(utts), 2)]
        p2_utts = [utts[i] for i in range(1, len(utts), 2)]
        if assume_persontokens:
            p1_utts = [remove_prefix(utt, p1_token) for utt in p1_utts]
            p2_utts = [remove_prefix(utt, p2_token) for utt in p2_utts]
        if dictionary is not None:
            persona_lines = [' '.join(dictionary.tokenize(pl)) for pl in persona_lines]
            p1_utts = [' '.join(dictionary.tokenize(utt)) for utt in p1_utts]
            p2_utts = [' '.join(dictionary.tokenize(utt)) for utt in p2_utts]
        self.persona_lines = [l.strip() for l in persona_lines if l.strip() != '']
        self.partner_utts = [l.strip() for l in p1_utts if l.strip() != '']
        self.own_utts = [l.strip() for l in p2_utts if l.strip() != '']