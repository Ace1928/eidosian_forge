import math
import sys
import warnings
from collections import Counter
from Bio import BiopythonDeprecationWarning
from Bio.Seq import Seq
def _get_letter_freqs(self, residue_num, all_records, letters, to_ignore, pseudo_count=0, e_freq_table=None, random_expected=None):
    """Determine the frequency of specific letters in the alignment (PRIVATE).

        Arguments:
         - residue_num - The number of the column we are getting frequencies
           from.
         - all_records - All of the SeqRecords in the alignment.
         - letters - The letters we are interested in getting the frequency
           for.
         - to_ignore - Letters we are specifically supposed to ignore.
         - pseudo_count - Optional argument specifying the Pseudo count (k)
           to add in order to prevent a frequency of 0 for a letter.
         - e_freq_table - An optional argument specifying a dictionary with
           the expected frequencies for each letter.
         - random_expected - Optional argument that specify the frequency to use
           when e_freq_table is not defined.

        This will calculate the frequencies of each of the specified letters
        in the alignment at the given frequency, and return this as a
        dictionary where the keys are the letters and the values are the
        frequencies. Pseudo count can be added to prevent a null frequency
        """
    freq_info = dict.fromkeys(letters, 0)
    total_count = 0
    gap_char = '-'
    if pseudo_count < 0:
        raise ValueError('Positive value required for pseudo_count, %s provided' % pseudo_count)
    for record in all_records:
        try:
            if record.seq[residue_num] not in to_ignore:
                weight = record.annotations.get('weight', 1.0)
                freq_info[record.seq[residue_num]] += weight
                total_count += weight
        except KeyError:
            raise ValueError('Residue %s not found in letters %s' % (record.seq[residue_num], letters)) from None
    if e_freq_table:
        for key in freq_info:
            if key != gap_char and key not in e_freq_table:
                raise ValueError('%s not found in expected frequency table' % key)
    if total_count == 0:
        for letter in freq_info:
            if freq_info[letter] != 0:
                raise ValueError('freq_info[letter] is not 0')
    else:
        for letter in freq_info:
            if pseudo_count and (random_expected or e_freq_table):
                if e_freq_table:
                    ajust_freq = e_freq_table[letter]
                else:
                    ajust_freq = random_expected
                ajusted_letter_count = freq_info[letter] + ajust_freq * pseudo_count
                ajusted_total = total_count + pseudo_count
                freq_info[letter] = ajusted_letter_count / ajusted_total
            else:
                freq_info[letter] = freq_info[letter] / total_count
    return freq_info