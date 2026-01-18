class OutSideTransaction(BzrError):
    _fmt = 'A transaction related operation was attempted after the transaction finished.'