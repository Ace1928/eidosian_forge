from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
class BuilderHandler(yaml_listener.EventHandler):
    """PyYAML event handler used to build objects.

  Maintains state information as it receives parse events so that object
  nesting is maintained.  Uses provided builder object to construct and
  assemble objects as it goes.

  As it receives events from the YAML parser, it builds a stack of data
  representing structural tokens.  As the scope of documents, mappings
  and sequences end, those token, value pairs are popped from the top of
  the stack so that the original scope can resume processing.

  A special case is made for the _KEY token.  It represents a temporary
  value which only occurs inside mappings.  It is immediately popped off
  the stack when it's associated value is encountered in the parse stream.
  It is necessary to do this because the YAML parser does not combine
  key and value information in to a single event.
  """

    def __init__(self, builder):
        """Initialization for builder handler.

    Args:
      builder: Instance of Builder class.

    Raises:
      ListenerConfigurationError when builder is not a Builder class.
    """
        if not isinstance(builder, Builder):
            raise yaml_errors.ListenerConfigurationError('Must provide builder of type yaml_listener.Builder')
        self._builder = builder
        self._stack = None
        self._top = None
        self._results = []

    def _Push(self, token, value):
        """Push values to stack at start of nesting.

    When a new object scope is beginning, will push the token (type of scope)
    along with the new objects value, the latter of which is provided through
    the various build methods of the builder.

    Args:
      token: Token indicating the type of scope which is being created; must
        belong to _TOKEN_VALUES.
      value: Value to associate with given token.  Construction of value is
        determined by the builder provided to this handler at construction.
    """
        self._top = (token, value)
        self._stack.append(self._top)

    def _Pop(self):
        """Pop values from stack at end of nesting.

    Called to indicate the end of a nested scope.

    Returns:
      Previously pushed value at the top of the stack.
    """
        assert self._stack != [] and self._stack is not None
        token, value = self._stack.pop()
        if self._stack:
            self._top = self._stack[-1]
        else:
            self._top = None
        return value

    def _HandleAnchor(self, event):
        """Handle anchor attached to event.

    Currently will raise an error if anchor is used.  Anchors are used to
    define a document wide tag to a given value (scalar, mapping or sequence).

    Args:
      event: Event which may have anchor property set.

    Raises:
      NotImplementedError if event attempts to use an anchor.
    """
        if hasattr(event, 'anchor') and event.anchor is not None:
            raise NotImplementedError('Anchors not supported in this handler')

    def _HandleValue(self, value):
        """Handle given value based on state of parser

    This method handles the various values that are created by the builder
    at the beginning of scope events (such as mappings and sequences) or
    when a scalar value is received.

    Method is called when handler receives a parser, MappingStart or
    SequenceStart.

    Args:
      value: Value received as scalar value or newly constructed mapping or
        sequence instance.

    Raises:
      InternalError if the building process encounters an unexpected token.
      This is an indication of an implementation error in BuilderHandler.
    """
        token, top_value = self._top
        if token == _TOKEN_KEY:
            key = self._Pop()
            mapping_token, mapping = self._top
            assert _TOKEN_MAPPING == mapping_token
            self._builder.MapTo(mapping, key, value)
        elif token == _TOKEN_MAPPING:
            self._Push(_TOKEN_KEY, value)
        elif token == _TOKEN_SEQUENCE:
            self._builder.AppendTo(top_value, value)
        elif token == _TOKEN_DOCUMENT:
            self._builder.InitializeDocument(top_value, value)
        else:
            raise yaml_errors.InternalError('Unrecognized builder token:\n%s' % token)

    def StreamStart(self, event, loader):
        """Initializes internal state of handler

    Args:
      event: Ignored.
    """
        assert self._stack is None
        self._stack = []
        self._top = None
        self._results = []

    def StreamEnd(self, event, loader):
        """Cleans up internal state of handler after parsing

    Args:
      event: Ignored.
    """
        assert self._stack == [] and self._top is None
        self._stack = None

    def DocumentStart(self, event, loader):
        """Build new document.

    Pushes new document on to stack.

    Args:
      event: Ignored.
    """
        assert self._stack == []
        self._Push(_TOKEN_DOCUMENT, self._builder.BuildDocument())

    def DocumentEnd(self, event, loader):
        """End of document.

    Args:
      event: Ignored.
    """
        assert self._top[0] == _TOKEN_DOCUMENT
        self._results.append(self._Pop())

    def Alias(self, event, loader):
        """Not implemented yet.

    Args:
      event: Ignored.
    """
        raise NotImplementedError('References not supported in this handler')

    def Scalar(self, event, loader):
        """Handle scalar value

    Since scalars are simple values that are passed directly in by the
    parser, handle like any value with no additional processing.

    Of course, key values will be handles specially.  A key value is recognized
    when the top token is _TOKEN_MAPPING.

    Args:
      event: Event containing scalar value.
    """
        self._HandleAnchor(event)
        if event.tag is None and self._top[0] != _TOKEN_MAPPING:
            try:
                tag = loader.resolve(yaml.nodes.ScalarNode, event.value, event.implicit)
            except IndexError:
                tag = loader.DEFAULT_SCALAR_TAG
        else:
            tag = event.tag
        if tag is None:
            value = event.value
        else:
            node = yaml.nodes.ScalarNode(tag, event.value, event.start_mark, event.end_mark, event.style)
            value = loader.construct_object(node)
        self._HandleValue(value)

    def SequenceStart(self, event, loader):
        """Start of sequence scope

    Create a new sequence from the builder and then handle in the context
    of its parent.

    Args:
      event: SequenceStartEvent generated by loader.
      loader: Loader that generated event.
    """
        self._HandleAnchor(event)
        token, parent = self._top
        if token == _TOKEN_KEY:
            token, parent = self._stack[-2]
        sequence = self._builder.BuildSequence(parent)
        self._HandleValue(sequence)
        self._Push(_TOKEN_SEQUENCE, sequence)

    def SequenceEnd(self, event, loader):
        """End of sequence.

    Args:
      event: Ignored
      loader: Ignored.
      """
        assert self._top[0] == _TOKEN_SEQUENCE
        end_object = self._Pop()
        top_value = self._top[1]
        self._builder.EndSequence(top_value, end_object)

    def MappingStart(self, event, loader):
        """Start of mapping scope.

    Create a mapping from builder and then handle in the context of its
    parent.

    Args:
      event: MappingStartEvent generated by loader.
      loader: Loader that generated event.
    """
        self._HandleAnchor(event)
        token, parent = self._top
        if token == _TOKEN_KEY:
            token, parent = self._stack[-2]
        mapping = self._builder.BuildMapping(parent)
        self._HandleValue(mapping)
        self._Push(_TOKEN_MAPPING, mapping)

    def MappingEnd(self, event, loader):
        """End of mapping

    Args:
      event: Ignored.
      loader: Ignored.
    """
        assert self._top[0] == _TOKEN_MAPPING
        end_object = self._Pop()
        top_value = self._top[1]
        self._builder.EndMapping(top_value, end_object)

    def GetResults(self):
        """Get results of document stream processing.

    This method can be invoked after fully parsing the entire YAML file
    to retrieve constructed contents of YAML file.  Called after EndStream.

    Returns:
      A tuple of all document objects that were parsed from YAML stream.

    Raises:
      InternalError if the builder stack is not empty by the end of parsing.
    """
        if self._stack is not None:
            raise yaml_errors.InternalError('Builder stack is not empty.')
        return tuple(self._results)