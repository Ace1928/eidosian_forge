import Bio.GenBank
def _base_count_line(self):
    """Output for the BASE COUNT line with base information (PRIVATE)."""
    output = ''
    if self.base_counts:
        output += Record.BASE_FORMAT % 'BASE COUNT  '
        count_parts = self.base_counts.split(' ')
        while '' in count_parts:
            count_parts.remove('')
        if len(count_parts) % 2 == 0:
            while len(count_parts) > 0:
                count_info = count_parts.pop(0)
                count_type = count_parts.pop(0)
                output += f'{count_info:>7} {count_type}'
        else:
            output += self.base_counts
        output += '\n'
    return output