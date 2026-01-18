from __future__ import annotations
import textwrap
from typing import TYPE_CHECKING, Literal, NamedTuple, cast
from streamlit import runtime
from streamlit.errors import StreamlitAPIException
from streamlit.proto import Block_pb2
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import WidgetArgs, WidgetCallback, WidgetKwargs
class FormMixin:

    @gather_metrics('form')
    def form(self, key: str, clear_on_submit: bool=False, *, border: bool=True) -> DeltaGenerator:
        """Create a form that batches elements together with a "Submit" button.

        A form is a container that visually groups other elements and
        widgets together, and contains a Submit button. When the form's
        Submit button is pressed, all widget values inside the form will be
        sent to Streamlit in a batch.

        To add elements to a form object, you can use "with" notation
        (preferred) or just call methods directly on the form. See
        examples below.

        Forms have a few constraints:

        * Every form must contain a ``st.form_submit_button``.
        * ``st.button`` and ``st.download_button`` cannot be added to a form.
        * Forms can appear anywhere in your app (sidebar, columns, etc),
          but they cannot be embedded inside other forms.
        * Within a form, the only widget that can have a callback function is
          ``st.form_submit_button``.

        Parameters
        ----------
        key : str
            A string that identifies the form. Each form must have its own
            key. (This key is not displayed to the user in the interface.)
        clear_on_submit : bool
            If True, all widgets inside the form will be reset to their default
            values after the user presses the Submit button. Defaults to False.
            (Note that Custom Components are unaffected by this flag, and
            will not be reset to their defaults on form submission.)
        border : bool
            Whether to show a border around the form. Defaults to True.

            .. note::
                Not showing a border can be confusing to viewers since interacting with a
                widget in the form will do nothing. You should only remove the border if
                there's another border (e.g. because of an expander) or the form is small
                (e.g. just a text input and a submit button).

        Examples
        --------
        Inserting elements using "with" notation:

        >>> import streamlit as st
        >>>
        >>> with st.form("my_form"):
        ...    st.write("Inside the form")
        ...    slider_val = st.slider("Form slider")
        ...    checkbox_val = st.checkbox("Form checkbox")
        ...
        ...    # Every form must have a submit button.
        ...    submitted = st.form_submit_button("Submit")
        ...    if submitted:
        ...        st.write("slider", slider_val, "checkbox", checkbox_val)
        ...
        >>> st.write("Outside the form")

        .. output::
           https://doc-form1.streamlit.app/
           height: 425px

        Inserting elements out of order:

        >>> import streamlit as st
        >>>
        >>> form = st.form("my_form")
        >>> form.slider("Inside the form")
        >>> st.slider("Outside the form")
        >>>
        >>> # Now add a submit button to the form:
        >>> form.form_submit_button("Submit")

        .. output::
           https://doc-form2.streamlit.app/
           height: 375px

        """
        from streamlit.elements.utils import check_session_state_rules
        if is_in_form(self.dg):
            raise StreamlitAPIException('Forms cannot be nested in other forms.')
        check_session_state_rules(default_value=None, key=key, writes_allowed=False)
        form_id = key
        ctx = get_script_run_ctx()
        if ctx is not None:
            new_form_id = form_id not in ctx.form_ids_this_run
            if new_form_id:
                ctx.form_ids_this_run.add(form_id)
            else:
                raise StreamlitAPIException(_build_duplicate_form_message(key))
        block_proto = Block_pb2.Block()
        block_proto.form.form_id = form_id
        block_proto.form.clear_on_submit = clear_on_submit
        block_proto.form.border = border
        block_dg = self.dg._block(block_proto)
        block_dg._form_data = FormData(form_id)
        return block_dg

    @gather_metrics('form_submit_button')
    def form_submit_button(self, label: str='Submit', help: str | None=None, on_click: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, type: Literal['primary', 'secondary']='secondary', disabled: bool=False, use_container_width: bool=False) -> bool:
        """Display a form submit button.

        When this button is clicked, all widget values inside the form will be
        sent to Streamlit in a batch.

        Every form must have a form_submit_button. A form_submit_button
        cannot exist outside a form.

        For more information about forms, check out our
        `blog post <https://blog.streamlit.io/introducing-submit-button-and-forms/>`_.

        Parameters
        ----------
        label : str
            A short label explaining to the user what this button is for.
            Defaults to "Submit".
        help : str or None
            A tooltip that gets displayed when the button is hovered over.
            Defaults to None.
        on_click : callable
            An optional callback invoked when this button is clicked.
        args : tuple
            An optional tuple of args to pass to the callback.
        kwargs : dict
            An optional dict of kwargs to pass to the callback.
        type : "secondary" or "primary"
            An optional string that specifies the button type. Can be "primary" for a
            button with additional emphasis or "secondary" for a normal button. Defaults
            to "secondary".
        disabled : bool
            An optional boolean, which disables the button if set to True. The
            default is False.
        use_container_width: bool
            An optional boolean, which makes the button stretch its width to match the parent container.


        Returns
        -------
        bool
            True if the button was clicked.
        """
        ctx = get_script_run_ctx()
        if type not in ['primary', 'secondary']:
            raise StreamlitAPIException(f'The type argument to st.button must be "primary" or "secondary". \nThe argument passed was "{type}".')
        return self._form_submit_button(label=label, help=help, on_click=on_click, args=args, kwargs=kwargs, type=type, disabled=disabled, use_container_width=use_container_width, ctx=ctx)

    def _form_submit_button(self, label: str='Submit', help: str | None=None, on_click: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, type: Literal['primary', 'secondary']='secondary', disabled: bool=False, use_container_width: bool=False, ctx: ScriptRunContext | None=None) -> bool:
        form_id = current_form_id(self.dg)
        submit_button_key = f'FormSubmitter:{form_id}-{label}'
        return self.dg._button(label=label, key=submit_button_key, help=help, is_form_submitter=True, on_click=on_click, args=args, kwargs=kwargs, type=type, disabled=disabled, use_container_width=use_container_width, ctx=ctx)

    @property
    def dg(self) -> DeltaGenerator:
        """Get our DeltaGenerator."""
        return cast('DeltaGenerator', self)