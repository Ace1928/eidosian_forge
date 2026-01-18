from parlai.core.dict import DictionaryAgent
from parlai.zoo.bert.build import download
from .helpers import VOCAB_PATH
import os
class BertDictionaryAgent(DictionaryAgent):
    """
    Allow to use the Torch Agent with the wordpiece dictionary of Hugging Face.
    """

    def __init__(self, opt):
        super().__init__(opt)
        download(opt['datapath'])
        vocab_path = os.path.join(opt['datapath'], 'models', 'bert_models', VOCAB_PATH)
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        self.start_token = '[CLS]'
        self.end_token = '[SEP]'
        self.null_token = '[PAD]'
        self.start_idx = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.end_idx = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.pad_idx = self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        self.tok2ind[self.start_token] = self.start_idx
        self.tok2ind[self.end_token] = self.end_idx
        self.tok2ind[self.null_token] = self.pad_idx
        self.ind2tok[self.start_idx] = self.start_token
        self.ind2tok[self.end_idx] = self.end_token
        self.ind2tok[self.pad_idx] = self.null_token

    def txt2vec(self, text, vec_type=list):
        tokens = self.tokenizer.tokenize(text)
        tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens_id

    def vec2txt(self, vec):
        if not isinstance(vec, list):
            idxs = [idx.item() for idx in vec.cpu()]
        else:
            idxs = vec
        toks = self.tokenizer.convert_ids_to_tokens(idxs)
        return ' '.join(toks)

    def act(self):
        return {}