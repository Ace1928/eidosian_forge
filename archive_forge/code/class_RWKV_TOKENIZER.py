import os, sys, time, random
class RWKV_TOKENIZER:
    table = None
    good = None
    wlen = None

    def __init__(self, file_name):
        self.vocab_size = 65525
        self.idx2token = {}
        sorted = []
        lines = open(file_name, 'r', encoding='utf-8').readlines()
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
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]
        for i in reversed(range(len(sorted))):
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes):
        src_len: int = len(src)
        tokens = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i:i + 1]
            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i:i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode('utf-8'))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def token_to_id(self, token):
        return self.token2idx[token]

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