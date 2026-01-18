import os, sys, time, random
class TRIE_TOKENIZER:

    def __init__(self, file_name):
        self.vocab_size = 65525
        self.idx2token = {}
        sorted = []
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode('utf-8') if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x
        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)
        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encodeBytes(self, src: bytes):
        idx: int = 0
        tokens = []
        while idx < len(src):
            _idx: int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert idx != _idx
            _, token = next(iter(values))
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src):
        return self.encodeBytes(src.encode('utf-8'))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.idx2token

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()