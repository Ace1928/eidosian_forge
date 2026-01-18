import Bio.GenBank
def _sequence_line(self):
    """Output for all of the sequence (PRIVATE)."""
    output = ''
    if self.sequence:
        cur_seq_pos = 0
        while cur_seq_pos < len(self.sequence):
            output += Record.SEQUENCE_FORMAT % str(cur_seq_pos + 1)
            for section in range(6):
                start_pos = cur_seq_pos + section * 10
                end_pos = start_pos + 10
                seq_section = self.sequence[start_pos:end_pos]
                output += f' {seq_section.lower()}'
                if end_pos > len(self.sequence):
                    break
            output += '\n'
            cur_seq_pos += 60
    return output