import sys  # Provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter. Documentation: https://docs.python.org/3/library/sys.html
import logging  # Provides a flexible framework for emitting log messages from Python programs. Documentation: https://docs.python.org/3/library/logging.html

from PyQt5.QtCore import (
    QUrl,
    Qt,
)  # Core non-GUI functionality. QUrl is used for working with URLs, and Qt provides namespace for miscellaneous identifiers. Documentation: https://doc.qt.io/qt-5/qtcore-module.html
from PyQt5.QtWidgets import (  # Provides a set of UI elements to create classic desktop-style user interfaces. Documentation: https://doc.qt.io/qt-5/qtwidgets-module.html
    QApplication,  # Manages application-wide resources and settings.
    QMainWindow,  # Provides a main application window with a menu bar, dock widgets, and a status bar.
    QToolBar,  # Provides a movable panel that contains a set of controls.
    QAction,  # Represents an action that can be added to widgets like toolbars and menus.
    QLineEdit,  # Provides a single-line text box.
    QStatusBar,  # Displays status information.
    QDialog,  # Provides a base class for dialog windows.
    QVBoxLayout,  # Lines up widgets vertically.
    QWidget,  # Base class for all UI objects.
)
from PyQt5.QtWebEngineWidgets import (
    QWebEngineView,  # Provides a widget that can render web content.
    QWebEnginePage,  # Provides a web page that can be rendered in a QWebEngineView.
)  # Provides a widget that can render web content. Documentation: https://doc.qt.io/qt-5/qtwebenginewidgets-module.html
from PyQt5.QtGui import (
    QKeySequence,
)  # Provides a way of handling key sequences. Documentation: https://doc.qt.io/qt-5/qtgui-module.html
from PyQt5.QtPrintSupport import (
    QPrintDialog,
    QPrinter,
)  # Provides support for printing. QPrintDialog is a dialog for specifying printer settings, and QPrinter is a paint device that paints on a printer. Documentation: https://doc.qt.io/qt-5/qtprintsupport-module.html

# Configure logging with DEBUG level and a specific format for timestamps, log levels, and messages
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CustomWebEnginePage(QWebEnginePage):
    def certificateError(self, certificateError):
        logging.error(f"SSL Error: {certificateError.errorDescription()}")
        # Decide whether to ignore the SSL error or not
        return certificateError.isOverridable()


class Browser(QMainWindow):
    def __init__(self):
        # Call the parent class (QMainWindow) constructor
        super().__init__()
        # Initialize the user interface
        self.init_ui()

    def init_ui(self):
        try:
            self.central_widget = QWidget()
            self.setCentralWidget(self.central_widget)
            self.layout = QVBoxLayout(self.central_widget)

            self.browser = QWebEngineView()
            self.browser.setPage(CustomWebEnginePage(self.browser))
            self.browser.setUrl(QUrl("http://www.google.com"))
            self.layout.addWidget(self.browser)

            self.create_navbar()
            self.create_menu()
            self.create_status_bar()

            # Connect signals to slots for URL changes and page load completion
            self.browser.urlChanged.connect(self.update_url)
            self.browser.loadFinished.connect(self.update_title)
            self.browser.loadFinished.connect(self.on_load_finished)
            self.browser.loadProgress.connect(self.on_load_progress)
            self.browser.loadStarted.connect(self.on_load_started)

            logging.info("UI initialized successfully.")
            self.showMaximized()
        except Exception as e:
            logging.error(f"Failed to initialize UI: {e}")

    def create_status_bar(self):
        try:
            # Create a status bar and set it for the main window
            self.status = QStatusBar()
            self.setStatusBar(self.status)

            # Connect browser load signals to status bar update slots
            self.browser.loadStarted.connect(self.on_load_started)
            self.browser.loadProgress.connect(self.on_load_progress)
            self.browser.loadFinished.connect(self.on_load_finished)
            logging.info("Status bar created successfully.")
        except Exception as e:
            logging.error(f"Failed to create status bar: {e}")

    def on_load_started(self):
        # Display a message in the status bar when page loading starts
        self.status.showMessage("Loading started...")

    def on_load_progress(self, progress):
        # Display the loading progress percentage in the status bar
        self.status.showMessage(f"Loading... {progress}%")

    def on_load_finished(self, ok):
        if not ok:
            logging.error("Failed to load the page.")
        self.status.showMessage("Loading finished.")

    def handle_ssl_errors(self, errors):
        for error in errors:
            logging.error(f"SSL Error: {error.errorString()}")

    def create_navbar(self):
        try:
            # Create a navigation toolbar and add it to the main window
            self.navbar = QToolBar("Navigation")
            self.addToolBar(self.navbar)

            # Create a back button with a shortcut and connect it to the browser's back function
            back_btn = QAction("⬅️ Back", self)
            back_btn.setShortcut(QKeySequence.Back)
            back_btn.triggered.connect(self.browser.back)
            self.navbar.addAction(back_btn)

            # Create a forward button with a shortcut and connect it to the browser's forward function
            forward_btn = QAction("➡️ Forward", self)
            forward_btn.setShortcut(QKeySequence.Forward)
            forward_btn.triggered.connect(self.browser.forward)
            self.navbar.addAction(forward_btn)

            # Create a reload button with a shortcut and connect it to the browser's reload function
            reload_btn = QAction("🔄 Reload", self)
            reload_btn.setShortcut(QKeySequence.Refresh)
            reload_btn.triggered.connect(self.browser.reload)
            self.navbar.addAction(reload_btn)

            # Create a home button with a shortcut and connect it to the navigate_home function
            home_btn = QAction("🏠 Home", self)
            home_btn.setShortcut("Ctrl+H")
            home_btn.triggered.connect(self.navigate_home)
            self.navbar.addAction(home_btn)

            # Create a URL bar and connect its returnPressed signal to the navigate_to_url function
            self.url_bar = QLineEdit()
            self.url_bar.returnPressed.connect(self.navigate_to_url)
            self.navbar.addWidget(self.url_bar)

            # Create a stop button with a shortcut and connect it to the browser's stop function
            stop_btn = QAction("⛔ Stop", self)
            stop_btn.setShortcut("Esc")
            stop_btn.triggered.connect(self.browser.stop)
            self.navbar.addAction(stop_btn)

            # Create a scrape button with a shortcut and connect it to the scrape_page function
            scrape_btn = QAction("🔍 Scrape", self)
            scrape_btn.setShortcut("Ctrl+S")
            scrape_btn.triggered.connect(self.scrape_page)
            self.navbar.addAction(scrape_btn)

            logging.info("Navigation bar created successfully.")
        except Exception as e:
            logging.error(f"Failed to create navigation bar: {e}")

    def create_menu(self):
        try:
            # Create a menu bar for the main window
            menubar = self.menuBar()

            # Create a File menu and add print and exit actions
            file_menu = menubar.addMenu("&File")
            print_action = QAction("🖨️ Print", self)
            print_action.setShortcut(QKeySequence.Print)
            print_action.triggered.connect(self.print_page)
            file_menu.addAction(print_action)

            exit_action = QAction("❌ Exit", self)
            exit_action.setShortcut(QKeySequence.Quit)
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)

            # Create a View menu and add zoom in, zoom out, and reset zoom actions
            view_menu = menubar.addMenu("&View")
            zoom_in_action = QAction("🔍➕ Zoom In", self)
            zoom_in_action.setShortcut(QKeySequence.ZoomIn)
            zoom_in_action.triggered.connect(
                lambda: self.browser.setZoomFactor(self.browser.zoomFactor() + 0.1)
            )
            view_menu.addAction(zoom_in_action)

            zoom_out_action = QAction("🔍➖ Zoom Out", self)
            zoom_out_action.setShortcut(QKeySequence.ZoomOut)
            zoom_out_action.triggered.connect(
                lambda: self.browser.setZoomFactor(self.browser.zoomFactor() - 0.1)
            )
            view_menu.addAction(zoom_out_action)

            reset_zoom_action = QAction("🔄 Reset Zoom", self)
            reset_zoom_action.setShortcut("Ctrl+0")
            reset_zoom_action.triggered.connect(lambda: self.browser.setZoomFactor(1.0))
            view_menu.addAction(reset_zoom_action)

            logging.info("Menu created successfully.")
        except Exception as e:
            logging.error(f"Failed to create menu: {e}")

    def navigate_home(self):
        try:
            # Set the browser URL to the home page (Google)
            self.browser.setUrl(QUrl("http://www.google.com"))
            logging.info("Navigated to home.")
        except Exception as e:
            logging.error(f"Failed to navigate home: {e}")

    def navigate_to_url(self):
        try:
            # Get the URL from the URL bar
            url = self.url_bar.text()
            # If the URL does not start with "http", prepend "http://"
            if not url.startswith("http"):
                url = "http://" + url
            # Set the browser URL to the entered URL
            self.browser.setUrl(QUrl(url))
            logging.info(f"Navigated to URL: {url}")
        except Exception as e:
            logging.error(f"Failed to navigate to URL: {e}")

    def update_url(self, q):
        try:
            # Update the URL bar with the current URL
            self.url_bar.setText(q.toString())
            logging.info(f"URL updated to: {q.toString()}")
        except Exception as e:
            logging.error(f"Failed to update URL: {e}")

    def update_title(self):
        try:
            # Get the page title and set it as the window title
            title = self.browser.page().title()
            self.setWindowTitle(f"{title} - Advanced Web Browser")
            logging.info(f"Title updated to: {title}")
        except Exception as e:
            logging.error(f"Failed to update title: {e}")

    def print_page(self):
        try:
            # Create a printer object and a print dialog
            printer = QPrinter()
            dialog = QPrintDialog(printer, self)
            # If the user accepts the print dialog, print the page
            if dialog.exec_() == QDialog.Accepted:
                self.browser.page().print(
                    printer,
                    lambda success: self.status.showMessage(
                        "Printing completed" if success else "Printing failed"
                    ),
                )
                logging.info("Print dialog executed.")
        except Exception as e:
            logging.error(f"Failed to print page: {e}")

    def scrape_page(self):
        try:
            # Initiate scraping the page content to HTML
            self.browser.page().toHtml(self.handle_scrape_result)
            logging.info("Scrape page initiated.")
        except Exception as e:
            logging.error(f"Failed to scrape page: {e}")

    def handle_scrape_result(self, html):
        try:
            # Log the scraped page content
            logging.info("Scraped page content")
            # Process the HTML content as needed (currently just printing it)
            print(html)
        except Exception as e:
            logging.error(f"Failed to handle scrape result: {e}")

    def closeEvent(self, event):
        try:
            # Accept the close event and log the application closure
            event.accept()
            logging.info("Application closed.")
        except Exception as e:
            logging.error(f"Failed to close application: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Browser()
    window.init_ui()
    window.show()
    sys.exit(app.exec_())
