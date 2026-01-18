from zope.interface import Attribute, Interface
class ISFTPFile(Interface):
    """
    This represents an open file on the server.  An object adhering to this
    interface should be returned from L{openFile}().
    """

    def close():
        """
        Close the file.

        This method returns nothing if the close succeeds immediately, or a
        Deferred that is called back when the close succeeds.
        """

    def readChunk(offset, length):
        """
        Read from the file.

        If EOF is reached before any data is read, raise EOFError.

        This method returns the data as a string, or a Deferred that is
        called back with same.

        @param offset: an integer that is the index to start from in the file.
        @param length: the maximum length of data to return.  The actual amount
        returned may less than this.  For normal disk files, however,
        this should read the requested number (up to the end of the file).
        """

    def writeChunk(offset, data):
        """
        Write to the file.

        This method returns when the write completes, or a Deferred that is
        called when it completes.

        @param offset: an integer that is the index to start from in the file.
        @param data: a string that is the data to write.
        """

    def getAttrs():
        """
        Return the attributes for the file.

        This method returns a dictionary in the same format as the attrs
        argument to L{openFile} or a L{Deferred} that is called back with same.
        """

    def setAttrs(attrs):
        """
        Set the attributes for the file.

        This method returns when the attributes are set or a Deferred that is
        called back when they are.

        @param attrs: a dictionary in the same format as the attrs argument to
        L{openFile}.
        """