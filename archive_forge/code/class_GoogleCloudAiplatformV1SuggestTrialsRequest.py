from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SuggestTrialsRequest(_messages.Message):
    """Request message for VizierService.SuggestTrials.

  Fields:
    clientId: Required. The identifier of the client that is requesting the
      suggestion. If multiple SuggestTrialsRequests have the same `client_id`,
      the service will return the identical suggested Trial if the Trial is
      pending, and provide a new Trial if the last suggested Trial was
      completed.
    contexts: Optional. This allows you to specify the "context" for a Trial;
      a context is a slice (a subspace) of the search space. Typical uses for
      contexts: 1) You are using Vizier to tune a server for best performance,
      but there's a strong weekly cycle. The context specifies the day-of-
      week. This allows Tuesday to generalize from Wednesday without assuming
      that everything is identical. 2) Imagine you're optimizing some medical
      treatment for people. As they walk in the door, you know certain facts
      about them (e.g. sex, weight, height, blood-pressure). Put that
      information in the context, and Vizier will adapt its suggestions to the
      patient. 3) You want to do a fair A/B test efficiently. Specify the "A"
      and "B" conditions as contexts, and Vizier will generalize between "A"
      and "B" conditions. If they are similar, this will allow Vizier to
      converge to the optimum faster than if "A" and "B" were separate
      Studies. NOTE: You can also enter contexts as REQUESTED Trials, e.g. via
      the CreateTrial() RPC; that's the asynchronous option where you don't
      need a close association between contexts and suggestions. NOTE: All the
      Parameters you set in a context MUST be defined in the Study. NOTE: You
      must supply 0 or $suggestion_count contexts. If you don't supply any
      contexts, Vizier will make suggestions from the full search space
      specified in the StudySpec; if you supply a full set of context, each
      suggestion will match the corresponding context. NOTE: A Context with no
      features set matches anything, and allows suggestions from the full
      search space. NOTE: Contexts MUST lie within the search space specified
      in the StudySpec. It's an error if they don't. NOTE: Contexts
      preferentially match ACTIVE then REQUESTED trials before new suggestions
      are generated. NOTE: Generation of suggestions involves a match between
      a Context and (optionally) a REQUESTED trial; if that match is not fully
      specified, a suggestion will be geneated in the merged subspace.
    suggestionCount: Required. The number of suggestions requested. It must be
      positive.
  """
    clientId = _messages.StringField(1)
    contexts = _messages.MessageField('GoogleCloudAiplatformV1TrialContext', 2, repeated=True)
    suggestionCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)