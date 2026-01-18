import re
from Bio import File
def _get_references(inl):
    references = []
    actref = ''
    for line in inl:
        if re.search('\\AREMARK   1', line):
            if re.search('\\AREMARK   1 REFERENCE', line):
                if actref != '':
                    actref = re.sub('\\s\\s+', ' ', actref)
                    if actref != ' ':
                        references.append(actref)
                    actref = ''
            else:
                actref += line[19:72].lower()
    if actref != '':
        actref = re.sub('\\s\\s+', ' ', actref)
        if actref != ' ':
            references.append(actref)
    return references