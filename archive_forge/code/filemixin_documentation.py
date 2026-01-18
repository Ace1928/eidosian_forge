
    Used to provide auxiliary methods to objects simulating files.
    Objects must implement write, and read if they are input files.
    Also they should implement close.

    Other methods you may wish to override:
    * flush()
    * seek(offset[, whence])
    * tell()
    * truncate([size])

    Attributes you may wish to provide:
    * closed
    * encoding (you should also respect that in write())
    * mode
    * newlines (hard to support)
    * softspace
    