class BackupTargetType:
    """
    Backup Target type.
    """
    VIRTUAL = 'Virtual'
    ' Denotes a virtual host '
    PHYSICAL = 'Physical'
    ' Denotes a physical host '
    FILESYSTEM = 'Filesystem'
    ' Denotes a file system (e.g. NAS) '
    DATABASE = 'Database'
    ' Denotes a database target '
    OBJECT = 'Object'
    ' Denotes an object based file system '
    VOLUME = 'Volume'
    ' Denotes a block storage volume '