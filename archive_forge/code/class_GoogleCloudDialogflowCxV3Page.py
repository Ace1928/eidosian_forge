from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3Page(_messages.Message):
    """A Dialogflow CX conversation (session) can be described and visualized
  as a state machine. The states of a CX session are represented by pages. For
  each flow, you define many pages, where your combined pages can handle a
  complete conversation on the topics the flow is designed for. At any given
  moment, exactly one page is the current page, the current page is considered
  active, and the flow associated with that page is considered active. Every
  flow has a special start page. When a flow initially becomes active, the
  start page page becomes the current page. For each conversational turn, the
  current page will either stay the same or transition to another page. You
  configure each page to collect information from the end-user that is
  relevant for the conversational state represented by the page. For more
  information, see the [Page
  guide](https://cloud.google.com/dialogflow/cx/docs/concept/page).

  Fields:
    advancedSettings: Hierarchical advanced settings for this page. The
      settings exposed at the lower level overrides the settings exposed at
      the higher level.
    description: The description of the page. The maximum length is 500
      characters.
    displayName: Required. The human-readable name of the page, unique within
      the flow.
    entryFulfillment: The fulfillment to call when the session is entering the
      page.
    eventHandlers: Handlers associated with the page to handle events such as
      webhook errors, no match or no input.
    form: The form associated with the page, used for collecting parameters
      relevant to the page.
    knowledgeConnectorSettings: Optional. Knowledge connector configuration.
    name: The unique identifier of the page. Required for the Pages.UpdatePage
      method. Pages.CreatePage populates the name automatically. Format:
      `projects//locations//agents//flows//pages/`.
    transitionRouteGroups: Ordered list of `TransitionRouteGroups` added to
      the page. Transition route groups must be unique within a page. If the
      page links both flow-level transition route groups and agent-level
      transition route groups, the flow-level ones will have higher priority
      and will be put before the agent-level ones. * If multiple transition
      routes within a page scope refer to the same intent, then the precedence
      order is: page's transition route -> page's transition route group ->
      flow's transition routes. * If multiple transition route groups within a
      page contain the same intent, then the first group in the ordered list
      takes precedence.
      Format:`projects//locations//agents//flows//transitionRouteGroups/` or
      `projects//locations//agents//transitionRouteGroups/` for agent-level
      groups.
    transitionRoutes: A list of transitions for the transition rules of this
      page. They route the conversation to another page in the same flow, or
      another flow. When we are in a certain page, the TransitionRoutes are
      evalauted in the following order: * TransitionRoutes defined in the page
      with intent specified. * TransitionRoutes defined in the transition
      route groups with intent specified. * TransitionRoutes defined in flow
      with intent specified. * TransitionRoutes defined in the transition
      route groups with intent specified. * TransitionRoutes defined in the
      page with only condition specified. * TransitionRoutes defined in the
      transition route groups with only condition specified.
  """
    advancedSettings = _messages.MessageField('GoogleCloudDialogflowCxV3AdvancedSettings', 1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    entryFulfillment = _messages.MessageField('GoogleCloudDialogflowCxV3Fulfillment', 4)
    eventHandlers = _messages.MessageField('GoogleCloudDialogflowCxV3EventHandler', 5, repeated=True)
    form = _messages.MessageField('GoogleCloudDialogflowCxV3Form', 6)
    knowledgeConnectorSettings = _messages.MessageField('GoogleCloudDialogflowCxV3KnowledgeConnectorSettings', 7)
    name = _messages.StringField(8)
    transitionRouteGroups = _messages.StringField(9, repeated=True)
    transitionRoutes = _messages.MessageField('GoogleCloudDialogflowCxV3TransitionRoute', 10, repeated=True)