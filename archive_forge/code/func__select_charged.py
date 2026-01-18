def _select_charged(self, aa_content):
    charged = {}
    for aa in charged_aas:
        charged[aa] = float(aa_content[aa])
    charged['Nterm'] = 1.0
    charged['Cterm'] = 1.0
    return charged