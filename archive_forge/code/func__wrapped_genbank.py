import Bio.GenBank
def _wrapped_genbank(information, indent, wrap_space=1, split_char=' '):
    """Write a line of GenBank info that can wrap over multiple lines (PRIVATE).

    This takes a line of information which can potentially wrap over
    multiple lines, and breaks it up with carriage returns and
    indentation so it fits properly into a GenBank record.

    Arguments:
     - information - The string holding the information we want
       wrapped in GenBank method.
     - indent - The indentation on the lines we are writing.
     - wrap_space - Whether or not to wrap only on spaces in the
       information.
     - split_char - A specific character to split the lines on. By default
       spaces are used.

    """
    info_length = Record.GB_LINE_LENGTH - indent
    if not information:
        return '.\n'
    if wrap_space:
        info_parts = information.split(split_char)
    else:
        cur_pos = 0
        info_parts = []
        while cur_pos < len(information):
            info_parts.append(information[cur_pos:cur_pos + info_length])
            cur_pos += info_length
    output_parts = []
    cur_part = ''
    for info_part in info_parts:
        if len(cur_part) + 1 + len(info_part) > info_length:
            if cur_part:
                if split_char != ' ':
                    cur_part += split_char
                output_parts.append(cur_part)
            cur_part = info_part
        elif cur_part == '':
            cur_part = info_part
        else:
            cur_part += split_char + info_part
    if cur_part:
        output_parts.append(cur_part)
    output_info = output_parts[0] + '\n'
    for output_part in output_parts[1:]:
        output_info += ' ' * indent + output_part + '\n'
    return output_info