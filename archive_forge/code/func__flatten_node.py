from . import static_tuple
def _flatten_node(node, reference_lists):
    """Convert a node into the serialized form.

    :param node: A tuple representing a node (key_tuple, value, references)
    :param reference_lists: Does this index have reference lists?
    :return: (string_key, flattened)
        string_key  The serialized key for referencing this node
        flattened   A string with the serialized form for the contents
    """
    if reference_lists:
        flattened_references = [b'\r'.join([b'\x00'.join(reference) for reference in ref_list]) for ref_list in node[3]]
    else:
        flattened_references = []
    string_key = b'\x00'.join(node[1])
    line = b'%s\x00%s\x00%s\n' % (string_key, b'\t'.join(flattened_references), node[2])
    return (string_key, line)