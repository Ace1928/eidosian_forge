@property
def _inconsistent_label(self):
    inconsistent = []
    if self.c3.direct_inconsistency:
        inconsistent.append('direct')
    if self.c3.bases_had_inconsistency:
        inconsistent.append('bases')
    return '+'.join(inconsistent) if inconsistent else 'no'