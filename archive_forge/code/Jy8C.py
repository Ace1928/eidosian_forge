import tkinter as tk
from tkinter import filedialog
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class UniversalGUIBuilder:
    """
    A class meticulously designed to construct and manage a GUI dynamically, allowing for runtime configuration
    and manipulation of widgets such as buttons, labels, and entries. This class supports saving and loading
    configurations to and from JSON files, providing a persistent state for the GUI layout and functionality.
    """

    def __init__(self, master: tk.Tk):
        """
        Initialize the UniversalGUIBuilder with a master window and setup the initial GUI components.

        :param master: The main tkinter window.
        :type master: tk.Tk
        """
        self.master = master
        self.master.title("Universal GUI Builder")
        self.frame = tk.Frame(self.master)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.config = {"buttons": [], "labels": [], "entries": []}

        self.menu_bar = tk.Menu(self.master)
        self.master.config(menu=self.menu_bar)

        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(
            label="Save Configuration", command=self.save_configuration
        )
        self.file_menu.add_command(
            label="Load Configuration", command=self.load_configuration
        )
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.add_widgets_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.add_widgets_menu.add_command(
            label="Add Button", command=lambda: self.add_widget("button")
        )
        self.add_widgets_menu.add_command(
            label="Add Label", command=lambda: self.add_widget("label")
        )
        self.add_widgets_menu.add_command(
            label="Add Entry", command=lambda: self.add_widget("entry")
        )
        self.menu_bar.add_cascade(label="Add Widgets", menu=self.add_widgets_menu)

        self.preview_button = tk.Button(
            self.frame, text="Preview GUI", command=self.preview_gui
        )
        self.preview_button.pack(side=tk.BOTTOM, pady=10)

    def add_widget(self, widget_type: str):
        """
        Opens a new window to add a widget of the specified type to the GUI configuration.

        :param widget_type: The type of widget to add ('button', 'label', 'entry').
        :type widget_type: str
        """
        new_window = tk.Toplevel(self.master)
        new_window.title(f"Add {widget_type.capitalize()}")

        label = tk.Label(new_window, text=f"Enter {widget_type} properties:")
        label.pack(side=tk.TOP, pady=10)

        name_label = tk.Label(new_window, text="Name:")
        name_label.pack(side=tk.TOP, pady=5)

        name_entry = tk.Entry(new_window)
        name_entry.pack(side=tk.TOP, pady=5)

        command_entry = None
        if widget_type != "label":
            command_label = tk.Label(new_window, text="Command (function name):")
            command_label.pack(side=tk.TOP, pady=5)

            command_entry = tk.Entry(new_window)
            command_entry.pack(side=tk.TOP, pady=5)

        save_button = tk.Button(
            new_window,
            text="Save",
            command=lambda: self.save_widget(
                new_window,
                widget_type,
                name_entry.get(),
                command_entry.get() if command_entry else "",
            ),
        )
        save_button.pack(side=tk.BOTTOM, pady=10)

    def save_widget(
        self, window: tk.Toplevel, widget_type: str, name: str, command: str = ""
    ):
        """
        Saves the widget configuration to the internal state and closes the configuration window.

        :param window: The window to close after saving.
        :type window: tk.Toplevel
        :param widget_type: The type of widget being saved.
        :type widget_type: str
        :param name: The name of the widget.
        :type name: str
        :param command: The command associated with the widget (if applicable).
        :type command: str
        """
        widget_config = {"name": name}
        if command:
            widget_config["command"] = command
        self.config[f"{widget_type}s"].append(widget_config)
        window.destroy()
        logging.info(f"Widget added: {widget_config}")

    def preview_gui(self):
        """
        Generates a preview of the GUI based on the current configuration.
        """
        preview_window = tk.Toplevel(self.master)
        preview_window.title("GUI Preview")

        for widget_type in self.config:
            for widget in self.config[widget_type]:
                if widget_type == "buttons":
                    tk.Button(
                        preview_window,
                        text=widget["name"],
                        command=lambda w=widget: print(f"Executing {w['command']}"),
                    ).pack(pady=5)
                elif widget_type == "labels":
                    tk.Label(preview_window, text=widget["name"]).pack(pady=5)
                elif widget_type == "entries":
                    tk.Entry(preview_window).pack(pady=5)

    def save_configuration(self):
        """
        Saves the current GUI configuration to a JSON file.
        """
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json")]
        )
        if file_path:
            with open(file_path, "w") as file:
                json.dump(self.config, file, indent=4)
            logging.info(f"Configuration saved to {file_path}")

    def load_configuration(self):
        """
        Loads a GUI configuration from a JSON file.
        """
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as file:
                self.config = json.load(file)
            logging.info(f"Configuration loaded from {file_path}")


def main():
    root = tk.Tk()
    app = UniversalGUIBuilder(root)
    root.mainloop()


if __name__ == "__main__":
    main()
