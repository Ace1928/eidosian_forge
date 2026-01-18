import Bio.GenBank
def _indent_genbank(information, indent):
    """Write out information with the specified indent (PRIVATE).

    Unlike _wrapped_genbank, this function makes no attempt to wrap
    lines -- it assumes that the information already has newlines in the
    appropriate places, and will add the specified indent to the start of
    each line.
    """
    info_parts = information.split('\n')
    output_info = info_parts[0] + '\n'
    for info_part in info_parts[1:]:
        output_info += ' ' * indent + info_part + '\n'
    return output_info