def __read_reference_start(record, line):
    reference = Reference()
    reference.number = line[1:3].strip()
    if line[1] == 'E':
        reference.citation = line[4:].strip()
    else:
        reference.authors = line[4:].strip()
    record.references.append(reference)