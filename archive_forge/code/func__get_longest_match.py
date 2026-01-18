from .. import osutils
def _get_longest_match(self, lines, pos):
    """Look at all matches for the current line, return the longest.

        :param lines: The lines we are matching against
        :param pos: The current location we care about
        :param locations: A list of lines that matched the current location.
            This may be None, but often we'll have already found matches for
            this line.
        :return: (start_in_self, start_in_lines, num_lines)
            All values are the offset in the list (aka the line number)
            If start_in_self is None, then we have no matches, and this line
            should be inserted in the target.
        """
    range_start = pos
    range_len = 0
    prev_locations = None
    max_pos = len(lines)
    matching = self._matching_lines
    while pos < max_pos:
        try:
            locations = matching[lines[pos]]
        except KeyError:
            pos += 1
            break
        if prev_locations is None:
            prev_locations = locations
            range_len = 1
            locations = None
        else:
            next_locations = locations.intersection([loc + 1 for loc in prev_locations])
            if next_locations:
                prev_locations = set(next_locations)
                range_len += 1
                locations = None
            else:
                break
        pos += 1
    if prev_locations is None:
        return (None, pos)
    smallest = min(prev_locations)
    return ((smallest - range_len + 1, range_start, range_len), pos)