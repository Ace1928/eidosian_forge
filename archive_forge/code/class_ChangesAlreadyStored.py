class ChangesAlreadyStored(CommandError):
    _fmt = 'Cannot store uncommitted changes because this branch already stores uncommitted changes.'