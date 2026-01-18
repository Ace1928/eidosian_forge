import re
from collections import defaultdict
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.PDBIO import Select, StructureIO
def _save_dict(self, out_file):
    key_lists = {}
    for key in self.dic:
        if key == 'data_':
            data_val = self.dic[key]
        else:
            s = re.split('\\.', key)
            if len(s) == 2:
                if s[0] in key_lists:
                    key_lists[s[0]].append(s[1])
                else:
                    key_lists[s[0]] = [s[1]]
            else:
                raise ValueError('Invalid key in mmCIF dictionary: ' + key)
    for key, key_list in key_lists.items():
        if key in mmcif_order:
            inds = []
            for i in key_list:
                try:
                    inds.append(mmcif_order[key].index(i))
                except ValueError:
                    inds.append(len(mmcif_order[key]))
            key_lists[key] = [k for _, k in sorted(zip(inds, key_list))]
    if data_val:
        out_file.write('data_' + data_val + '\n#\n')
    for key, key_list in key_lists.items():
        sample_val = self.dic[key + '.' + key_list[0]]
        n_vals = len(sample_val)
        for i in key_list:
            val = self.dic[key + '.' + i]
            if isinstance(sample_val, list) and (isinstance(val, str) or len(val) != n_vals) or (isinstance(sample_val, str) and isinstance(val, list)):
                raise ValueError('Inconsistent list sizes in mmCIF dictionary: ' + key + '.' + i)
        if isinstance(sample_val, str) or (isinstance(sample_val, list) and len(sample_val) == 1):
            m = 0
            for i in key_list:
                if len(i) > m:
                    m = len(i)
            for i in key_list:
                if isinstance(sample_val, str):
                    value_no_list = self.dic[key + '.' + i]
                else:
                    value_no_list = self.dic[key + '.' + i][0]
                out_file.write('{k: <{width}}'.format(k=key + '.' + i, width=len(key) + m + 4) + self._format_mmcif_col(value_no_list, len(value_no_list)) + '\n')
        elif isinstance(sample_val, list):
            out_file.write('loop_\n')
            col_widths = {}
            for i in key_list:
                out_file.write(key + '.' + i + '\n')
                col_widths[i] = 0
                for val in self.dic[key + '.' + i]:
                    len_val = len(val)
                    if self._requires_quote(val) and (not self._requires_newline(val)):
                        len_val += 2
                    if len_val > col_widths[i]:
                        col_widths[i] = len_val
            for i in range(n_vals):
                for col in key_list:
                    out_file.write(self._format_mmcif_col(self.dic[key + '.' + col][i], col_widths[col] + 1))
                out_file.write('\n')
        else:
            raise ValueError('Invalid type in mmCIF dictionary: ' + str(type(sample_val)))
        out_file.write('#\n')