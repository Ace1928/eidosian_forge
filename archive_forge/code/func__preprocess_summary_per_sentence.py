import nltk
import os
import re
import itertools
import collections
import pkg_resources
def _preprocess_summary_per_sentence(self, summary):
    """
        Preprocessing (truncate text if enable, tokenization, stemming if enable, lowering) of a summary by sentences

        Args:
          summary: string of the summary

        Returns:
          Return the preprocessed summary (string)
        """
    sentences = Rouge.split_into_sentences(summary, self.ensure_compatibility)
    if self.limit_length:
        final_sentences = []
        current_len = 0
        if self.length_limit_type == 'words':
            for sentence in sentences:
                tokens = sentence.strip().split()
                tokens_len = len(tokens)
                if current_len + tokens_len < self.length_limit:
                    sentence = ' '.join(tokens)
                    final_sentences.append(sentence)
                    current_len += tokens_len
                else:
                    sentence = ' '.join(tokens[:self.length_limit - current_len])
                    final_sentences.append(sentence)
                    break
        elif self.length_limit_type == 'bytes':
            for sentence in sentences:
                sentence = sentence.strip()
                sentence_len = len(sentence)
                if current_len + sentence_len < self.length_limit:
                    final_sentences.append(sentence)
                    current_len += sentence_len
                else:
                    sentence = sentence[:self.length_limit - current_len]
                    final_sentences.append(sentence)
                    break
        sentences = final_sentences
    final_sentences = []
    for sentence in sentences:
        sentence = Rouge.REMOVE_CHAR_PATTERN.sub(' ', sentence.lower()).strip()
        if self.ensure_compatibility:
            tokens = self.tokenize_text(Rouge.KEEP_CANNOT_IN_ONE_WORD.sub('_cannot_', sentence))
        else:
            tokens = self.tokenize_text(Rouge.REMOVE_CHAR_PATTERN.sub(' ', sentence))
        if self.stemming:
            self.stem_tokens(tokens)
        if self.ensure_compatibility:
            sentence = Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub('cannot', ' '.join(tokens))
        else:
            sentence = ' '.join(tokens)
        final_sentences.append(sentence)
    return final_sentences