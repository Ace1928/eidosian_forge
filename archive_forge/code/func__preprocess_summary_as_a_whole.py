import nltk
import os
import re
import itertools
import collections
import pkg_resources
def _preprocess_summary_as_a_whole(self, summary):
    """
        Preprocessing (truncate text if enable, tokenization, stemming if enable, lowering) of a summary as a whole

        Args:
          summary: string of the summary

        Returns:
          Return the preprocessed summary (string)
        """
    sentences = Rouge.split_into_sentences(summary, self.ensure_compatibility)
    if self.limit_length:
        if self.length_limit_type == 'words':
            summary = ' '.join(sentences)
            all_tokens = summary.split()
            summary = ' '.join(all_tokens[:self.length_limit])
        elif self.length_limit_type == 'bytes':
            summary = ''
            current_len = 0
            for sentence in sentences:
                sentence = sentence.strip()
                sentence_len = len(sentence)
                if current_len + sentence_len < self.length_limit:
                    if current_len != 0:
                        summary += ' '
                    summary += sentence
                    current_len += sentence_len
                else:
                    if current_len > 0:
                        summary += ' '
                    summary += sentence[:self.length_limit - current_len]
                    break
    else:
        summary = ' '.join(sentences)
    summary = Rouge.REMOVE_CHAR_PATTERN.sub(' ', summary.lower()).strip()
    if self.ensure_compatibility:
        tokens = self.tokenize_text(Rouge.KEEP_CANNOT_IN_ONE_WORD.sub('_cannot_', summary))
    else:
        tokens = self.tokenize_text(Rouge.REMOVE_CHAR_PATTERN.sub(' ', summary))
    if self.stemming:
        self.stem_tokens(tokens)
    if self.ensure_compatibility:
        preprocessed_summary = [Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub('cannot', ' '.join(tokens))]
    else:
        preprocessed_summary = [' '.join(tokens)]
    return preprocessed_summary