from pygments.lexer import RegexLexer, words, include
from pygments.token import Comment, Keyword, Name, Number, Punctuation, \
class RideLexer(RegexLexer):
    """
    For `Ride <https://docs.wavesplatform.com/en/ride/about-ride.html>`_
    source code.

    .. versionadded:: 2.6
    """
    name = 'Ride'
    aliases = ['ride']
    filenames = ['*.ride']
    mimetypes = ['text/x-ride']
    validName = "[a-zA-Z_][a-zA-Z0-9_\\']*"
    builtinOps = ('||', '|', '>=', '>', '==', '!', '=', '<=', '<', '::', ':+', ':', '!=', '/', '.', '=>', '-', '+', '*', '&&', '%', '++')
    globalVariablesName = ('NOALG', 'MD5', 'SHA1', 'SHA224', 'SHA256', 'SHA384', 'SHA512', 'SHA3224', 'SHA3256', 'SHA3384', 'SHA3512', 'nil', 'this', 'unit', 'height', 'lastBlock', 'Buy', 'Sell', 'CEILING', 'FLOOR', 'DOWN', 'HALFDOWN', 'HALFEVEN', 'HALFUP', 'UP')
    typesName = ('Unit', 'Int', 'Boolean', 'ByteVector', 'String', 'Address', 'Alias', 'Transfer', 'AssetPair', 'DataEntry', 'Order', 'Transaction', 'GenesisTransaction', 'PaymentTransaction', 'ReissueTransaction', 'BurnTransaction', 'MassTransferTransaction', 'ExchangeTransaction', 'TransferTransaction', 'SetAssetScriptTransaction', 'InvokeScriptTransaction', 'IssueTransaction', 'LeaseTransaction', 'LeaseCancelTransaction', 'CreateAliasTransaction', 'SetScriptTransaction', 'SponsorFeeTransaction', 'DataTransaction', 'WriteSet', 'AttachedPayment', 'ScriptTransfer', 'TransferSet', 'ScriptResult', 'Invocation', 'Asset', 'BlockInfo', 'Issue', 'Reissue', 'Burn', 'NoAlg', 'Md5', 'Sha1', 'Sha224', 'Sha256', 'Sha384', 'Sha512', 'Sha3224', 'Sha3256', 'Sha3384', 'Sha3512', 'BinaryEntry', 'BooleanEntry', 'IntegerEntry', 'StringEntry', 'List', 'Ceiling', 'Down', 'Floor', 'HalfDown', 'HalfEven', 'HalfUp', 'Up')
    functionsName = ('fraction', 'size', 'toBytes', 'take', 'drop', 'takeRight', 'dropRight', 'toString', 'isDefined', 'extract', 'throw', 'getElement', 'value', 'cons', 'toUtf8String', 'toInt', 'indexOf', 'lastIndexOf', 'split', 'parseInt', 'parseIntValue', 'keccak256', 'blake2b256', 'sha256', 'sigVerify', 'toBase58String', 'fromBase58String', 'toBase64String', 'fromBase64String', 'transactionById', 'transactionHeightById', 'getInteger', 'getBoolean', 'getBinary', 'getString', 'addressFromPublicKey', 'addressFromString', 'addressFromRecipient', 'assetBalance', 'wavesBalance', 'getIntegerValue', 'getBooleanValue', 'getBinaryValue', 'getStringValue', 'addressFromStringValue', 'assetInfo', 'rsaVerify', 'checkMerkleProof', 'median', 'valueOrElse', 'valueOrErrorMessage', 'contains', 'log', 'pow', 'toBase16String', 'fromBase16String', 'blockInfoByHeight', 'transferTransactionById')
    reservedWords = words(('match', 'case', 'else', 'func', 'if', 'let', 'then', '@Callable', '@Verifier'), suffix='\\b')
    tokens = {'root': [('#.*', Comment.Single), ('\\s+', Text), ('"', String, 'doublequote'), ("utf8\\'", String, 'utf8quote'), ("base(58|64|16)\\'", String, 'singlequote'), (reservedWords, Keyword.Reserved), ('\\{-#.*?#-\\}', Keyword.Reserved), ('FOLD<\\d+>', Keyword.Reserved), (words(typesName), Keyword.Type), (words(builtinOps, prefix='\\(', suffix='\\)'), Name.Function), (words(builtinOps), Name.Function), (words(globalVariablesName), Name.Function), (words(functionsName), Name.Function), include('numbers'), (validName, Name.Variable), ('[,()\\[\\]{}]', Punctuation)], 'doublequote': [('\\\\u[0-9a-fA-F]{4}', String.Escape), ('\\\\[nrfvb\\\\"]', String.Escape), ('[^"]', String), ('"', String, '#pop')], 'utf8quote': [('\\\\u[0-9a-fA-F]{4}', String.Escape), ("\\\\[nrfvb\\\\\\']", String.Escape), ("[^\\']", String), ("\\'", String, '#pop')], 'singlequote': [("[^\\']", String), ("\\'", String, '#pop')], 'numbers': [('_?\\d+', Number.Integer)]}