import re
from Bio import File
def _get_journal(inl):
    journal = ''
    for line in inl:
        if re.search('\\AJRNL', line):
            journal += line[19:72].lower()
    journal = re.sub('\\s\\s+', ' ', journal)
    return journal